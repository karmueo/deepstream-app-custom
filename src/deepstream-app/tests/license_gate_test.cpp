#include "license_gate_internal.hpp"

#include <glib.h>
#include <glib/gstdio.h>
#include <openssl/evp.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{

int failures = 0;

#define CHECK(condition)                                                       \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            g_printerr("CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                       #condition);                                             \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

struct TestKey
{
    EVP_PKEY *key = nullptr;
    std::array<unsigned char, 32> public_key{};

    TestKey()
    {
        EVP_PKEY_CTX *context = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
        CHECK(context != nullptr);
        CHECK(EVP_PKEY_keygen_init(context) == 1);
        CHECK(EVP_PKEY_keygen(context, &key) == 1);
        EVP_PKEY_CTX_free(context);
        size_t length = public_key.size();
        CHECK(EVP_PKEY_get_raw_public_key(key, public_key.data(), &length) == 1);
        CHECK(length == public_key.size());
    }

    ~TestKey()
    {
        EVP_PKEY_free(key);
    }
};

std::vector<unsigned char> sign(EVP_PKEY *key, const std::string &payload)
{
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    CHECK(context != nullptr);
    CHECK(EVP_DigestSignInit(context, nullptr, nullptr, nullptr, key) == 1);
    size_t length = 0;
    CHECK(EVP_DigestSign(context, nullptr, &length,
                         reinterpret_cast<const unsigned char *>(payload.data()),
                         payload.size()) == 1);
    std::vector<unsigned char> signature(length);
    CHECK(EVP_DigestSign(context, signature.data(), &length,
                         reinterpret_cast<const unsigned char *>(payload.data()),
                         payload.size()) == 1);
    signature.resize(length);
    EVP_MD_CTX_free(context);
    return signature;
}

std::string base64(const unsigned char *data, size_t size)
{
    gchar *encoded = g_base64_encode(data, size);
    std::string result(encoded);
    g_free(encoded);
    return result;
}

std::string make_payload(const std::string &fingerprint,
                         const std::string &product = "deepstream-app-custom")
{
    return "{\"customer\":\"Acme\",\"device_fingerprint\":\"" +
           fingerprint +
           "\",\"issued_at\":\"2026-09-11T00:00:00Z\","
           "\"license_id\":\"72c48243-1b7f-46ea-a543-538e8b9f4f8b\","
           "\"perpetual\":true,\"product\":\"" +
           product + "\",\"schema\":1}";
}

std::string make_envelope(EVP_PKEY *key, const std::string &payload,
                          const std::string &key_id = "test-key")
{
    std::vector<unsigned char> signature = sign(key, payload);
    return "{\"key_id\":\"" + key_id + "\",\"payload\":\"" +
           base64(reinterpret_cast<const unsigned char *>(payload.data()),
                  payload.size()) +
           "\",\"schema\":1,\"signature\":\"" +
           base64(signature.data(), signature.size()) + "\"}\n";
}

std::string temporary_file(const std::string &contents)
{
    gchar *name = nullptr;
    GError *error = nullptr;
    int descriptor = g_file_open_tmp("license-gate-test-XXXXXX", &name, &error);
    CHECK(descriptor >= 0);
    if (descriptor >= 0)
        close(descriptor);
    CHECK(error == nullptr);
    g_clear_error(&error);
    CHECK(g_file_set_contents(name, contents.data(), contents.size(), &error));
    CHECK(error == nullptr);
    g_clear_error(&error);
    std::string result(name ? name : "");
    g_free(name);
    return result;
}

void remove_file(const std::string &path)
{
    if (!path.empty())
        CHECK(g_remove(path.c_str()) == 0);
}

void test_fingerprint()
{
    std::string lower;
    std::string upper;
    std::string error;
    CHECK(ds_license::fingerprint_serial("  abcd1234\0\n", &lower, &error));
    CHECK(ds_license::fingerprint_serial("ABCD1234", &upper, &error));
    CHECK(lower == upper);
    CHECK(lower.size() == 71);
    CHECK(lower.rfind("sha256:", 0) == 0);
    CHECK(!ds_license::fingerprint_serial("bad", &upper, &error));
    CHECK(!ds_license::fingerprint_serial("", &upper, &error));
}

void test_serial_fallback()
{
    gchar *directory = g_dir_make_tmp("license-serial-test-XXXXXX", nullptr);
    CHECK(directory != nullptr);
    std::string missing = std::string(directory) + "/missing";
    std::string fallback = std::string(directory) + "/serial";
    CHECK(g_file_set_contents(fallback.c_str(), "1422825049051\0", 14, nullptr));
    std::string serial;
    std::string error;
    CHECK(ds_license::find_serial({missing, fallback}, &serial, &error));
    CHECK(serial == "1422825049051");
    remove_file(fallback);
    CHECK(g_rmdir(directory) == 0);
    g_free(directory);
}

void test_validation()
{
    TestKey key;
    std::string fingerprint;
    std::string error;
    CHECK(ds_license::fingerprint_serial("1422825049051", &fingerprint,
                                         &error));
    DsLicenseInfo info{};

    std::string valid_payload = make_payload(fingerprint);
    std::string valid_path = temporary_file(make_envelope(key.key, valid_payload));
    CHECK(ds_license::validate_for_fingerprint(valid_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) == DS_LICENSE_OK);
    CHECK(std::string(info.customer) == "Acme");

    CHECK(ds_license::validate_for_fingerprint(
              "/tmp/deepstream-license-does-not-exist", fingerprint,
              key.public_key, "test-key", &info) == DS_LICENSE_NOT_FOUND);
    CHECK(ds_license::validate_for_fingerprint(valid_path, fingerprint,
                                               key.public_key, "unknown-key",
                                               &info) == DS_LICENSE_KEY_ERROR);

    std::string other_fingerprint;
    CHECK(ds_license::fingerprint_serial("1422825049052", &other_fingerprint,
                                         &error));
    CHECK(ds_license::validate_for_fingerprint(valid_path, other_fingerprint,
                                               key.public_key, "test-key",
                                               &info) ==
          DS_LICENSE_DEVICE_MISMATCH);

    std::string product_path = temporary_file(
        make_envelope(key.key, make_payload(fingerprint, "other-product")));
    CHECK(ds_license::validate_for_fingerprint(product_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) ==
          DS_LICENSE_PRODUCT_MISMATCH);

    std::string invalid_fingerprint_path = temporary_file(
        make_envelope(key.key, make_payload("sha256:not-a-fingerprint")));
    CHECK(ds_license::validate_for_fingerprint(
              invalid_fingerprint_path, fingerprint, key.public_key,
              "test-key", &info) == DS_LICENSE_FORMAT_ERROR);

    std::string tampered = make_envelope(key.key, valid_payload);
    size_t signature_position = tampered.find("\"signature\":\"");
    CHECK(signature_position != std::string::npos);
    signature_position += std::strlen("\"signature\":\"");
    tampered[signature_position] =
        tampered[signature_position] == 'A' ? 'B' : 'A';
    std::string tampered_path = temporary_file(tampered);
    CHECK(ds_license::validate_for_fingerprint(tampered_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) ==
          DS_LICENSE_SIGNATURE_ERROR);

    std::string payload_tampered = make_envelope(key.key, valid_payload);
    size_t payload_position = payload_tampered.find("\"payload\":\"");
    CHECK(payload_position != std::string::npos);
    payload_position += std::strlen("\"payload\":\"");
    payload_tampered[payload_position] =
        payload_tampered[payload_position] == 'A' ? 'B' : 'A';
    std::string payload_tampered_path = temporary_file(payload_tampered);
    CHECK(ds_license::validate_for_fingerprint(
              payload_tampered_path, fingerprint, key.public_key, "test-key",
              &info) == DS_LICENSE_SIGNATURE_ERROR);

    std::string malformed_path = temporary_file("{\"schema\":1");
    CHECK(ds_license::validate_for_fingerprint(malformed_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) == DS_LICENSE_FORMAT_ERROR);

    std::string truncated_path = temporary_file(
        make_envelope(key.key, valid_payload).substr(0, 64));
    CHECK(ds_license::validate_for_fingerprint(truncated_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) == DS_LICENSE_FORMAT_ERROR);

    std::string oversized_path = temporary_file(std::string(16 * 1024 + 1, 'x'));
    CHECK(ds_license::validate_for_fingerprint(oversized_path, fingerprint,
                                               key.public_key, "test-key",
                                               &info) == DS_LICENSE_FORMAT_ERROR);

    remove_file(valid_path);
    remove_file(product_path);
    remove_file(invalid_fingerprint_path);
    remove_file(tampered_path);
    remove_file(payload_tampered_path);
    remove_file(malformed_path);
    remove_file(truncated_path);
    remove_file(oversized_path);
}

} // namespace

int main()
{
    test_fingerprint();
    test_serial_fallback();
    test_validation();
    if (failures != 0)
        g_printerr("%d license test checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
