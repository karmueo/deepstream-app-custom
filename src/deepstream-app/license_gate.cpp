#include "license_gate.h"
#include "license_gate_internal.hpp"
#include "license_public_key.h"

#include <glib.h>
#include <json-glib/json-glib.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr size_t kMaxLicenseSize = 16 * 1024;
constexpr size_t kMaxPayloadSize = 8 * 1024;
constexpr const char *kProduct = "deepstream-app-custom";
constexpr const char *kFingerprintPrefix = "sha256:";
constexpr const char *kFingerprintDomain = "deepstream-app-custom/v1";

const std::vector<std::string> kSerialPaths = {
    "/proc/device-tree/serial-number",
    "/sys/devices/soc0/serial_number",
    "/sys/bus/soc/devices/soc0/serial_number",
};

void clear_info(DsLicenseInfo *info)
{
    if (info)
        std::memset(info, 0, sizeof(*info));
}

void set_error(DsLicenseInfo *info, DsLicenseStatus status,
               const std::string &message)
{
    if (!info)
        return;
    info->status = status;
    g_strlcpy(info->error, message.c_str(), sizeof(info->error));
}

bool decode_hex_key(const char *hex, std::array<unsigned char, 32> *key)
{
    if (!hex || !key || std::strlen(hex) != key->size() * 2)
        return false;

    auto hex_value = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    };

    for (size_t i = 0; i < key->size(); ++i)
    {
        int high = hex_value(hex[i * 2]);
        int low = hex_value(hex[i * 2 + 1]);
        if (high < 0 || low < 0)
            return false;
        (*key)[i] = static_cast<unsigned char>((high << 4) | low);
    }
    return true;
}

bool valid_base64(const std::string &value)
{
    if (value.empty() || value.size() % 4 != 0)
        return false;

    size_t padding = 0;
    if (value.back() == '=')
        ++padding;
    if (value.size() > 1 && value[value.size() - 2] == '=')
        ++padding;

    for (size_t i = 0; i < value.size(); ++i)
    {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (std::isalnum(c) || c == '+' || c == '/')
            continue;
        if (c == '=' && i >= value.size() - padding)
            continue;
        return false;
    }
    return true;
}

bool decode_base64(const std::string &value, size_t maximum,
                   std::vector<unsigned char> *decoded)
{
    if (!decoded || !valid_base64(value))
        return false;
    gsize length = 0;
    guchar *data = g_base64_decode(value.c_str(), &length);
    if (!data || length > maximum)
    {
        g_free(data);
        return false;
    }
    decoded->assign(data, data + length);
    g_free(data);
    return true;
}

bool json_string(JsonObject *object, const char *name, std::string *value)
{
    if (!object || !name || !value || !json_object_has_member(object, name))
        return false;
    JsonNode *node = json_object_get_member(object, name);
    if (!node || !JSON_NODE_HOLDS_VALUE(node) ||
        json_node_get_value_type(node) != G_TYPE_STRING)
        return false;
    const char *text = json_node_get_string(node);
    if (!text)
        return false;
    *value = text;
    return true;
}

bool json_int(JsonObject *object, const char *name, gint64 *value)
{
    if (!object || !name || !value || !json_object_has_member(object, name))
        return false;
    JsonNode *node = json_object_get_member(object, name);
    if (!node || !JSON_NODE_HOLDS_VALUE(node) ||
        json_node_get_value_type(node) != G_TYPE_INT64)
        return false;
    *value = json_node_get_int(node);
    return true;
}

bool json_bool(JsonObject *object, const char *name, bool *value)
{
    if (!object || !name || !value || !json_object_has_member(object, name))
        return false;
    JsonNode *node = json_object_get_member(object, name);
    if (!node || !JSON_NODE_HOLDS_VALUE(node) ||
        json_node_get_value_type(node) != G_TYPE_BOOLEAN)
        return false;
    *value = json_node_get_boolean(node);
    return true;
}

bool parse_object(const unsigned char *data, size_t length, JsonParser **parser,
                  JsonObject **object, std::string *error)
{
    if (!data || !parser || !object)
        return false;
    GError *parse_error = nullptr;
    *parser = json_parser_new();
    if (!json_parser_load_from_data(*parser,
                                    reinterpret_cast<const gchar *>(data),
                                    static_cast<gssize>(length), &parse_error))
    {
        if (error)
            *error = parse_error ? parse_error->message : "JSON parse failed";
        g_clear_error(&parse_error);
        g_object_unref(*parser);
        *parser = nullptr;
        return false;
    }
    JsonNode *root = json_parser_get_root(*parser);
    if (!root || !JSON_NODE_HOLDS_OBJECT(root))
    {
        if (error)
            *error = "JSON root must be an object";
        g_object_unref(*parser);
        *parser = nullptr;
        return false;
    }
    *object = json_node_get_object(root);
    return true;
}

bool constant_time_equal(const std::string &left, const std::string &right)
{
    return left.size() == right.size() &&
           CRYPTO_memcmp(left.data(), right.data(), left.size()) == 0;
}

bool valid_fingerprint(const std::string &fingerprint)
{
    if (fingerprint.size() != DS_LICENSE_FINGERPRINT_SIZE - 1 ||
        fingerprint.rfind(kFingerprintPrefix, 0) != 0)
        return false;
    return std::all_of(fingerprint.begin() + std::strlen(kFingerprintPrefix),
                       fingerprint.end(), [](unsigned char c) {
                           return (c >= '0' && c <= '9') ||
                                  (c >= 'a' && c <= 'f');
                       });
}

bool verify_ed25519(const std::array<unsigned char, 32> &public_key,
                    const std::vector<unsigned char> &payload,
                    const std::vector<unsigned char> &signature)
{
    if (signature.size() != 64)
        return false;
    EVP_PKEY *key = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, nullptr, public_key.data(), public_key.size());
    if (!key)
        return false;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    bool valid = context &&
                 EVP_DigestVerifyInit(context, nullptr, nullptr, nullptr, key) == 1 &&
                 EVP_DigestVerify(context, signature.data(), signature.size(),
                                  payload.data(), payload.size()) == 1;
    EVP_MD_CTX_free(context);
    EVP_PKEY_free(key);
    return valid;
}

std::string trim_serial(std::string value)
{
    auto removable = [](unsigned char c) {
        return c == '\0' || std::isspace(c);
    };
    while (!value.empty() && removable(value.front()))
        value.erase(value.begin());
    while (!value.empty() && removable(value.back()))
        value.pop_back();
    return value;
}

bool valid_serial(const std::string &serial)
{
    if (serial.size() < 8 || serial.size() > 64)
        return false;
    return std::all_of(serial.begin(), serial.end(), [](unsigned char c) {
        return std::isalnum(c) != 0;
    });
}

std::string read_device_model()
{
    gchar *data = nullptr;
    gsize length = 0;
    if (!g_file_get_contents("/proc/device-tree/model", &data, &length, nullptr))
        return "unknown";
    std::string model(data, length);
    g_free(data);
    model = trim_serial(model);
    if (model.empty() || model.size() > 256)
        return "unknown";
    return model;
}

std::string random_request_id()
{
    std::array<unsigned char, 16> bytes{};
    if (RAND_bytes(bytes.data(), bytes.size()) != 1)
        return {};
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes)
    {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 0x0f]);
    }
    return result;
}

} // namespace

namespace ds_license
{

bool fingerprint_serial(const std::string &serial_input,
                        std::string *fingerprint, std::string *error)
{
    if (!fingerprint)
        return false;
    std::string serial = trim_serial(serial_input);
    if (!valid_serial(serial))
    {
        if (error)
            *error = "Jetson serial number is missing or invalid";
        return false;
    }
    std::transform(serial.begin(), serial.end(), serial.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    std::string input(kFingerprintDomain);
    input.push_back('\0');
    input.append(serial);
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_length = 0;
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    bool ok = context && EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context, input.data(), input.size()) == 1 &&
              EVP_DigestFinal_ex(context, digest.data(), &digest_length) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_length != 32)
    {
        if (error)
            *error = "Unable to calculate device fingerprint";
        return false;
    }

    static const char digits[] = "0123456789abcdef";
    fingerprint->assign(kFingerprintPrefix);
    for (size_t i = 0; i < digest_length; ++i)
    {
        fingerprint->push_back(digits[digest[i] >> 4]);
        fingerprint->push_back(digits[digest[i] & 0x0f]);
    }
    return true;
}

bool find_serial(const std::vector<std::string> &paths, std::string *serial,
                 std::string *error)
{
    if (!serial)
        return false;
    for (const std::string &path : paths)
    {
        if (!g_file_test(path.c_str(), G_FILE_TEST_EXISTS))
            continue;
        gchar *data = nullptr;
        gsize length = 0;
        GError *read_error = nullptr;
        if (!g_file_get_contents(path.c_str(), &data, &length, &read_error))
        {
            if (error)
                *error = "Unable to read Jetson serial number from " + path;
            g_clear_error(&read_error);
            return false;
        }
        std::string candidate(data, length);
        g_free(data);
        candidate = trim_serial(candidate);
        if (!valid_serial(candidate))
        {
            if (error)
                *error = "Invalid Jetson serial number in " + path;
            return false;
        }
        *serial = candidate;
        return true;
    }
    if (error)
        *error = "Jetson serial number is unavailable";
    return false;
}

DsLicenseStatus validate_for_fingerprint(
    const std::string &license_path, const std::string &fingerprint,
    const std::array<unsigned char, 32> &public_key,
    const std::string &expected_key_id, DsLicenseInfo *info)
{
    clear_info(info);
    if (info)
        g_strlcpy(info->device_fingerprint, fingerprint.c_str(),
                  sizeof(info->device_fingerprint));

    if (!g_file_test(license_path.c_str(), G_FILE_TEST_EXISTS))
    {
        set_error(info, DS_LICENSE_NOT_FOUND,
                  "License file not found: " + license_path);
        return DS_LICENSE_NOT_FOUND;
    }

    gchar *file_data = nullptr;
    gsize file_length = 0;
    GError *read_error = nullptr;
    if (!g_file_get_contents(license_path.c_str(), &file_data, &file_length,
                             &read_error))
    {
        std::string message = read_error ? read_error->message
                                         : "Unable to read license file";
        g_clear_error(&read_error);
        set_error(info, DS_LICENSE_IO_ERROR, message);
        return DS_LICENSE_IO_ERROR;
    }
    if (file_length == 0 || file_length > kMaxLicenseSize)
    {
        g_free(file_data);
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "License file is empty or exceeds 16 KiB");
        return DS_LICENSE_FORMAT_ERROR;
    }

    JsonParser *envelope_parser = nullptr;
    JsonObject *envelope = nullptr;
    std::string parse_error;
    if (!parse_object(reinterpret_cast<unsigned char *>(file_data), file_length,
                      &envelope_parser, &envelope, &parse_error))
    {
        g_free(file_data);
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "Invalid license envelope: " + parse_error);
        return DS_LICENSE_FORMAT_ERROR;
    }
    g_free(file_data);

    gint64 schema = 0;
    std::string key_id;
    std::string payload_base64;
    std::string signature_base64;
    if (!json_int(envelope, "schema", &schema) || schema != 1 ||
        !json_string(envelope, "key_id", &key_id) ||
        !json_string(envelope, "payload", &payload_base64) ||
        !json_string(envelope, "signature", &signature_base64))
    {
        g_object_unref(envelope_parser);
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "License envelope fields are missing or invalid");
        return DS_LICENSE_FORMAT_ERROR;
    }
    if (key_id != expected_key_id)
    {
        g_object_unref(envelope_parser);
        set_error(info, DS_LICENSE_KEY_ERROR, "Unknown license signing key");
        return DS_LICENSE_KEY_ERROR;
    }

    std::vector<unsigned char> payload;
    std::vector<unsigned char> signature;
    if (!decode_base64(payload_base64, kMaxPayloadSize, &payload) ||
        !decode_base64(signature_base64, 64, &signature) ||
        signature.size() != 64)
    {
        g_object_unref(envelope_parser);
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "License payload or signature encoding is invalid");
        return DS_LICENSE_FORMAT_ERROR;
    }
    if (!verify_ed25519(public_key, payload, signature))
    {
        g_object_unref(envelope_parser);
        set_error(info, DS_LICENSE_SIGNATURE_ERROR,
                  "License signature verification failed");
        return DS_LICENSE_SIGNATURE_ERROR;
    }
    g_object_unref(envelope_parser);

    JsonParser *payload_parser = nullptr;
    JsonObject *payload_object = nullptr;
    if (!parse_object(payload.data(), payload.size(), &payload_parser,
                      &payload_object, &parse_error))
    {
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "Invalid signed license payload: " + parse_error);
        return DS_LICENSE_FORMAT_ERROR;
    }

    gint64 payload_schema = 0;
    std::string license_id;
    std::string product;
    std::string licensed_fingerprint;
    std::string customer;
    std::string issued_at;
    bool perpetual = false;
    bool fields_ok = json_int(payload_object, "schema", &payload_schema) &&
                     payload_schema == 1 &&
                     json_string(payload_object, "license_id", &license_id) &&
                     json_string(payload_object, "product", &product) &&
                     json_string(payload_object, "device_fingerprint",
                                 &licensed_fingerprint) &&
                     json_string(payload_object, "customer", &customer) &&
                     json_string(payload_object, "issued_at", &issued_at) &&
                     json_bool(payload_object, "perpetual", &perpetual) &&
                     perpetual && !license_id.empty() &&
                     license_id.size() < DS_LICENSE_ID_SIZE &&
                     !customer.empty() &&
                     customer.size() < DS_LICENSE_CUSTOMER_SIZE &&
                     !issued_at.empty() && issued_at.size() <= 64 &&
                     valid_fingerprint(licensed_fingerprint);
    if (!fields_ok)
    {
        g_object_unref(payload_parser);
        set_error(info, DS_LICENSE_FORMAT_ERROR,
                  "Signed license payload fields are missing or invalid");
        return DS_LICENSE_FORMAT_ERROR;
    }
    if (product != kProduct)
    {
        g_object_unref(payload_parser);
        set_error(info, DS_LICENSE_PRODUCT_MISMATCH,
                  "License is for a different product");
        return DS_LICENSE_PRODUCT_MISMATCH;
    }
    if (!constant_time_equal(licensed_fingerprint, fingerprint))
    {
        g_object_unref(payload_parser);
        set_error(info, DS_LICENSE_DEVICE_MISMATCH,
                  "License is bound to a different Jetson device");
        return DS_LICENSE_DEVICE_MISMATCH;
    }

    if (info)
    {
        info->status = DS_LICENSE_OK;
        g_strlcpy(info->license_id, license_id.c_str(), sizeof(info->license_id));
        g_strlcpy(info->customer, customer.c_str(), sizeof(info->customer));
        info->error[0] = '\0';
    }
    g_object_unref(payload_parser);
    return DS_LICENSE_OK;
}

} // namespace ds_license

DsLicenseStatus ds_license_get_device_fingerprint(
    char fingerprint[DS_LICENSE_FINGERPRINT_SIZE],
    char error[DS_LICENSE_ERROR_SIZE])
{
    if (fingerprint)
        fingerprint[0] = '\0';
    if (error)
        error[0] = '\0';
    if (!fingerprint)
        return DS_LICENSE_DEVICE_UNAVAILABLE;

    std::string serial;
    std::string message;
    std::string result;
    if (!ds_license::find_serial(kSerialPaths, &serial, &message) ||
        !ds_license::fingerprint_serial(serial, &result, &message))
    {
        if (error)
            g_strlcpy(error, message.c_str(), DS_LICENSE_ERROR_SIZE);
        return DS_LICENSE_DEVICE_UNAVAILABLE;
    }
    g_strlcpy(fingerprint, result.c_str(), DS_LICENSE_FINGERPRINT_SIZE);
    return DS_LICENSE_OK;
}

DsLicenseStatus ds_license_validate(const char *license_path,
                                    DsLicenseInfo *info)
{
    clear_info(info);
    char fingerprint[DS_LICENSE_FINGERPRINT_SIZE] = {0};
    char error[DS_LICENSE_ERROR_SIZE] = {0};
    DsLicenseStatus status =
        ds_license_get_device_fingerprint(fingerprint, error);
    if (status != DS_LICENSE_OK)
    {
        set_error(info, status, error);
        return status;
    }

    std::array<unsigned char, 32> public_key{};
    if (!decode_hex_key(DS_LICENSE_PUBLIC_KEY_HEX, &public_key))
    {
        set_error(info, DS_LICENSE_KEY_ERROR,
                  "Embedded license public key is invalid");
        return DS_LICENSE_KEY_ERROR;
    }
    const char *path = (license_path && license_path[0])
                           ? license_path
                           : DS_LICENSE_DEFAULT_PATH;
    return ds_license::validate_for_fingerprint(path, fingerprint, public_key,
                                                DS_LICENSE_KEY_ID, info);
}

DsLicenseStatus ds_license_write_request(const char *output_path,
                                         const char *app_version,
                                         DsLicenseInfo *info)
{
    clear_info(info);
    if (!output_path || !output_path[0])
    {
        set_error(info, DS_LICENSE_IO_ERROR,
                  "License request output path is empty");
        return DS_LICENSE_IO_ERROR;
    }

    char fingerprint[DS_LICENSE_FINGERPRINT_SIZE] = {0};
    char device_error[DS_LICENSE_ERROR_SIZE] = {0};
    DsLicenseStatus status =
        ds_license_get_device_fingerprint(fingerprint, device_error);
    if (status != DS_LICENSE_OK)
    {
        set_error(info, status, device_error);
        return status;
    }
    std::string request_id = random_request_id();
    if (request_id.empty())
    {
        set_error(info, DS_LICENSE_IO_ERROR,
                  "Unable to generate license request identifier");
        return DS_LICENSE_IO_ERROR;
    }

    JsonBuilder *builder = json_builder_new();
    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "schema");
    json_builder_add_int_value(builder, 1);
    json_builder_set_member_name(builder, "request_id");
    json_builder_add_string_value(builder, request_id.c_str());
    json_builder_set_member_name(builder, "product");
    json_builder_add_string_value(builder, kProduct);
    json_builder_set_member_name(builder, "device_fingerprint");
    json_builder_add_string_value(builder, fingerprint);
    json_builder_set_member_name(builder, "device_model");
    json_builder_add_string_value(builder, read_device_model().c_str());
    json_builder_set_member_name(builder, "app_version");
    json_builder_add_string_value(builder,
                                  app_version ? app_version : "unknown");
    json_builder_end_object(builder);

    JsonGenerator *generator = json_generator_new();
    JsonNode *root = json_builder_get_root(builder);
    json_generator_set_root(generator, root);
    json_generator_set_pretty(generator, TRUE);
    gchar *request_data = json_generator_to_data(generator, nullptr);
    GError *write_error = nullptr;
    gboolean written = g_file_set_contents(output_path, request_data, -1,
                                           &write_error);
    std::string message = write_error ? write_error->message : "";
    g_clear_error(&write_error);
    g_free(request_data);
    json_node_free(root);
    g_object_unref(generator);
    g_object_unref(builder);

    if (!written)
    {
        set_error(info, DS_LICENSE_IO_ERROR,
                  "Unable to write license request: " + message);
        return DS_LICENSE_IO_ERROR;
    }
    if (info)
    {
        info->status = DS_LICENSE_OK;
        g_strlcpy(info->device_fingerprint, fingerprint,
                  sizeof(info->device_fingerprint));
    }
    return DS_LICENSE_OK;
}

const char *ds_license_status_name(DsLicenseStatus status)
{
    switch (status)
    {
    case DS_LICENSE_OK:
        return "ok";
    case DS_LICENSE_DEVICE_UNAVAILABLE:
        return "device-unavailable";
    case DS_LICENSE_NOT_FOUND:
        return "not-found";
    case DS_LICENSE_IO_ERROR:
        return "io-error";
    case DS_LICENSE_FORMAT_ERROR:
        return "format-error";
    case DS_LICENSE_KEY_ERROR:
        return "key-error";
    case DS_LICENSE_SIGNATURE_ERROR:
        return "signature-error";
    case DS_LICENSE_PRODUCT_MISMATCH:
        return "product-mismatch";
    case DS_LICENSE_DEVICE_MISMATCH:
        return "device-mismatch";
    default:
        return "unknown";
    }
}
