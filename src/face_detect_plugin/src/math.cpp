#include "face_core.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace face {
namespace {
cv::Mat blob(const cv::Mat &image, float mean, float stddev) {
    if (image.empty() || image.type() != CV_8UC3 || !(stddev > 0))
        throw std::runtime_error("invalid BGR image or normalization");
    int dims[] = {1, 3, image.rows, image.cols};
    cv::Mat out(4, dims, CV_32F);
    const size_t plane = image.total();
    for (int y = 0; y < image.rows; ++y) {
        const auto *row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x)
            for (int c = 0; c < 3; ++c)
                out.ptr<float>()[c * plane + y * image.cols + x] =
                    (row[x][2 - c] - mean) / stddev;
    }
    return out;
}
}
DetectionInput detector_input(const cv::Mat &image) {
    if (image.empty()) throw std::runtime_error("empty image");
    const double ratio = double(image.rows) / image.cols;
    const int h = ratio > 1 ? 640 : std::max(1, int(640 * ratio));
    const int w = ratio > 1 ? std::max(1, int(640 / ratio)) : 640;
    cv::Mat padded = cv::Mat::zeros(640, 640, CV_8UC3);
    cv::resize(image, padded(cv::Rect(0, 0, w, h)), {w, h});
    return {blob(padded, 127.5f, 128.f), float(h) / image.rows};
}
cv::Mat recognition_input(const cv::Mat &image, float mean, float stddev) {
    return blob(image, mean, stddev);
}
cv::Mat alignment_matrix(const std::array<cv::Point2f, 5> &points) {
    const std::array<cv::Point2d, 5> dst = {{{38.2946, 51.6963}, {73.5318, 51.5014},
        {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}}};
    cv::Mat a(10, 4, CV_64F, cv::Scalar(0)), b(10, 1, CV_64F), x;
    double variance = 0;
    cv::Point2d center;
    for (auto p : points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            throw std::runtime_error("invalid landmarks");
        center += cv::Point2d(p);
    }
    center *= .2;
    for (int i = 0; i < 5; ++i) {
        const double px = points[i].x, py = points[i].y;
        variance += cv::norm(cv::Point2d(points[i]) - center);
        a.at<double>(2 * i, 0) = px; a.at<double>(2 * i, 1) = -py;
        a.at<double>(2 * i, 2) = 1;
        a.at<double>(2 * i + 1, 0) = py; a.at<double>(2 * i + 1, 1) = px;
        a.at<double>(2 * i + 1, 3) = 1;
        b.at<double>(2 * i) = dst[i].x; b.at<double>(2 * i + 1) = dst[i].y;
    }
    if (variance < 1e-6 || !cv::solve(a, b, x, cv::DECOMP_SVD))
        throw std::runtime_error("degenerate landmarks");
    return (cv::Mat_<double>(2, 3) << x.at<double>(0), -x.at<double>(1), x.at<double>(2),
            x.at<double>(1), x.at<double>(0), x.at<double>(3));
}
cv::Mat align_face(const cv::Mat &image, const Detection &f) {
    cv::Mat out;
    cv::warpAffine(image, out, alignment_matrix(f.points), {112, 112},
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return out;
}
std::vector<Detection> nms(std::vector<Detection> faces, float threshold) {
    std::stable_sort(faces.begin(), faces.end(),
                     [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto &f : faces) {
        bool suppress = false;
        for (const auto &k : kept) {
            const float w = std::max(0.f, std::min(f.box.x + f.box.width, k.box.x + k.box.width) -
                                      std::max(f.box.x, k.box.x) + 1);
            const float h = std::max(0.f, std::min(f.box.y + f.box.height, k.box.y + k.box.height) -
                                      std::max(f.box.y, k.box.y) + 1);
            const float inter = w * h;
            const float denom = (f.box.width + 1) * (f.box.height + 1) +
                                (k.box.width + 1) * (k.box.height + 1) - inter;
            if (denom > 0 && inter / denom > threshold) { suppress = true; break; }
        }
        if (!suppress) kept.push_back(f);
    }
    return kept;
}
std::vector<Detection> decode_scrfd(const std::vector<std::vector<float>> &outputs,
                                    float scale, float threshold, float nms_threshold) {
    if (outputs.size() != 9 || !(scale > 0))
        throw std::runtime_error("SCRFD requires nine outputs and a positive scale");
    std::vector<Detection> faces;
    for (int level = 0; level < 3; ++level) {
        const int stride = 8 << level, width = 640 / stride, count = width * width * 2;
        const auto &scores = outputs[level];
        const auto &boxes = outputs[level + 3];
        const auto &points = outputs[level + 6];
        if (scores.size() != size_t(count) || boxes.size() != size_t(count * 4) ||
            points.size() != size_t(count * 10))
            throw std::runtime_error("unexpected SCRFD output shape/order");
        for (int i = 0; i < count; ++i) {
            if (scores[i] < threshold) continue;
            const float cx = (i / 2 % width) * stride, cy = (i / 2 / width) * stride;
            const float x1 = (cx - boxes[4 * i] * stride) / scale;
            const float y1 = (cy - boxes[4 * i + 1] * stride) / scale;
            const float x2 = (cx + boxes[4 * i + 2] * stride) / scale;
            const float y2 = (cy + boxes[4 * i + 3] * stride) / scale;
            if (x2 <= x1 || y2 <= y1) continue;
            Detection f;
            f.box = {x1, y1, x2 - x1, y2 - y1}; f.score = scores[i];
            for (int j = 0; j < 5; ++j)
                f.points[j] = {(cx + points[10 * i + 2 * j] * stride) / scale,
                               (cy + points[10 * i + 2 * j + 1] * stride) / scale};
            faces.push_back(f);
        }
    }
    return nms(std::move(faces), nms_threshold);
}
bool normalize(std::vector<float> &v) {
    double norm = 0;
    for (float x : v) {
        if (!std::isfinite(x)) return false;
        norm += double(x) * x;
    }
    if (norm < 1e-20) return false;
    norm = std::sqrt(norm);
    for (float &x : v) x = float(x / norm);
    return true;
}
Match match(const std::vector<float> &embedding, const std::vector<Sample> &gallery,
            float threshold) {
    Match result;
    float best = -2;
    auto query = embedding;
    if (!normalize(query)) return result;
    for (const auto &s : gallery) {
        if (s.embedding.size() != query.size()) continue;
        float similarity = std::inner_product(query.begin(), query.end(), s.embedding.begin(), 0.f);
        if (!std::isfinite(similarity)) continue;
        similarity = std::clamp(similarity, -1.f, 1.f);
        if (similarity > best) {
            best = similarity; result.similarity = similarity;
            result.matched = similarity >= threshold;
            result.person = result.matched ? s.person : 0;
            result.name = result.matched ? s.name : "陌生人";
        }
    }
    return result;
}
} // namespace face
