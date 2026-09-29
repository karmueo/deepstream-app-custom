#include "face_core.hpp"
#include <cstring>
#include <filesystem>
#include <sqlite3.h>
#include <stdexcept>

namespace face {
namespace {
void check(int result, sqlite3 *db) {
    if (result != SQLITE_OK && result != SQLITE_DONE && result != SQLITE_ROW)
        throw std::runtime_error(sqlite3_errmsg(db));
}
class Statement {
public:
    Statement(sqlite3 *db, const char *sql) : db_(db) {
        check(sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr), db);
    }
    ~Statement() { if (stmt_) sqlite3_finalize(stmt_); }
    sqlite3_stmt *get() const { return stmt_; }
    int step() { const int rc = sqlite3_step(stmt_); check(rc, db_); return rc; }
private:
    sqlite3 *db_;
    sqlite3_stmt *stmt_ = nullptr;
};
}
struct Gallery::Impl {
    sqlite3 *db = nullptr;
    bool readonly = true;
    ~Impl() { if (db) sqlite3_close(db); }
};
Gallery::Gallery(const std::string &path, bool readonly) : p_(std::make_unique<Impl>()) {
    p_->readonly = readonly;
    if (!readonly) std::filesystem::create_directories(std::filesystem::absolute(path).parent_path());
    int flags = readonly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    int rc = sqlite3_open_v2(path.c_str(), &p_->db, flags, nullptr);
    if (rc != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(p_->db));
    sqlite3_busy_timeout(p_->db, 5000);
    if (!readonly) {
        const char *sql =
            "PRAGMA journal_mode=WAL;PRAGMA foreign_keys=ON;"
            "CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY,name TEXT NOT NULL UNIQUE);"
            "CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY,person INTEGER REFERENCES "
            "people(id),source TEXT,hash TEXT,model TEXT,embedding BLOB,UNIQUE(person,hash,model));";
        char *error = nullptr;
        rc = sqlite3_exec(p_->db, sql, nullptr, nullptr, &error);
        if (rc != SQLITE_OK) {
            std::string message = error ? error : sqlite3_errmsg(p_->db);
            sqlite3_free(error); throw std::runtime_error(message);
        }
    }
}
Gallery::~Gallery() = default;
int64_t Gallery::add_person(const std::string &name) {
    if (p_->readonly || name.empty()) throw std::runtime_error("invalid person name or read-only gallery");
    Statement insert(p_->db, "INSERT OR IGNORE INTO people(name) VALUES(?)");
    check(sqlite3_bind_text(insert.get(), 1, name.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    insert.step();
    Statement select(p_->db, "SELECT id FROM people WHERE name=?");
    check(sqlite3_bind_text(select.get(), 1, name.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    if (select.step() != SQLITE_ROW) throw std::runtime_error("person insert failed");
    return sqlite3_column_int64(select.get(), 0);
}
bool Gallery::add_sample(int64_t person, const std::string &source, const std::string &hash,
                         const std::string &model, const std::vector<float> &embedding) {
    if (p_->readonly) throw std::runtime_error("read-only gallery");
    auto normalized = embedding;
    if (!normalize(normalized)) throw std::runtime_error("invalid gallery embedding");
    Statement insert(p_->db, "INSERT OR IGNORE INTO samples(person,source,hash,model,embedding) "
                           "VALUES(?,?,?,?,?)");
    check(sqlite3_bind_int64(insert.get(), 1, person), p_->db);
    check(sqlite3_bind_text(insert.get(), 2, source.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    check(sqlite3_bind_text(insert.get(), 3, hash.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    check(sqlite3_bind_text(insert.get(), 4, model.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    check(sqlite3_bind_blob(insert.get(), 5, normalized.data(),
          int(normalized.size() * sizeof(float)), SQLITE_TRANSIENT), p_->db);
    insert.step();
    return sqlite3_changes(p_->db) > 0;
}
std::vector<Sample> Gallery::samples(const std::string &model) const {
    Statement query(p_->db, "SELECT p.id,p.name,s.embedding FROM samples s JOIN people p "
                           "ON p.id=s.person WHERE s.model=? ORDER BY p.id,s.id");
    check(sqlite3_bind_text(query.get(), 1, model.c_str(), -1, SQLITE_TRANSIENT), p_->db);
    std::vector<Sample> samples;
    while (query.step() == SQLITE_ROW) {
        const void *blob = sqlite3_column_blob(query.get(), 2);
        const int bytes = sqlite3_column_bytes(query.get(), 2);
        if (!blob || bytes <= 0 || bytes % sizeof(float))
            throw std::runtime_error("corrupt gallery embedding");
        Sample sample;
        sample.person = sqlite3_column_int64(query.get(), 0);
        sample.name = reinterpret_cast<const char *>(sqlite3_column_text(query.get(), 1));
        sample.embedding.resize(bytes / sizeof(float));
        std::memcpy(sample.embedding.data(), blob, bytes);
        if (!normalize(sample.embedding)) throw std::runtime_error("corrupt gallery embedding");
        samples.push_back(std::move(sample));
    }
    return samples;
}
} // namespace face
