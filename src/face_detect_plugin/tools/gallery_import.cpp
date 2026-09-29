#include "face_core.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
int main(int argc, char **argv) {
    try {
        std::string config_file, photos, database;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--config" || arg == "--photos" || arg == "--database") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--config") config_file = value;
                else if (arg == "--photos") photos = value;
                else database = value;
            } else throw std::runtime_error("usage: face-gallery-import --config FILE [--photos DIR] [--database FILE]");
        }
        if (config_file.empty()) throw std::runtime_error("--config is required");
        const auto config = face::load_config(config_file);
        if (photos.empty()) photos = (fs::absolute(config_file).parent_path() / "data").string();
        if (database.empty()) database = config.gallery_file;
        if (database.empty()) throw std::runtime_error("gallery-file is empty; specify --database");
        if (!fs::is_directory(photos)) throw std::runtime_error("photos directory not found");
        face::Pipeline pipeline(config, 0);
        pipeline.warmup();
        face::Gallery gallery(database, false);
        std::vector<fs::path> people;
        for (const auto &entry : fs::directory_iterator(photos))
            if (entry.is_directory()) people.push_back(entry.path());
        std::sort(people.begin(), people.end());
        int imported = 0, duplicates = 0, skipped = 0;
        for (const auto &person_dir : people) {
            int64_t person_id = 0;
            std::vector<fs::path> images;
            for (const auto &entry : fs::recursive_directory_iterator(person_dir)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                    ext == ".bmp" || ext == ".webp") images.push_back(entry.path());
            }
            std::sort(images.begin(), images.end());
            for (const auto &path : images) {
                std::string reason;
                try {
                    cv::Mat image = cv::imread(path.string());
                    if (image.empty()) reason = "无法读取图片";
                    else {
                        auto faces = pipeline.analyze(image);
                        if (faces.empty()) reason = "未检测到人脸";
                        else if (faces.size() != 1) reason = "包含多张人脸";
                        else if (faces[0].box.width < 32 || faces[0].box.height < 32)
                            reason = "人脸小于 32 像素";
                        else if (faces[0].embedding.empty()) reason = "无有效人脸特征";
                        else {
                            if (!person_id) person_id = gallery.add_person(person_dir.filename().string());
                            if (gallery.add_sample(person_id, path.string(), face::sha256_file(path.string()),
                                                   pipeline.model_id(), faces[0].embedding)) ++imported;
                            else ++duplicates;
                        }
                    }
                } catch (const std::exception &e) { reason = e.what(); }
                if (!reason.empty()) { ++skipped; std::cout << "跳过 " << path << ": " << reason << '\n'; }
            }
        }
        std::cout << "导入 " << imported << "，重复 " << duplicates << "，跳过 " << skipped << '\n';
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
