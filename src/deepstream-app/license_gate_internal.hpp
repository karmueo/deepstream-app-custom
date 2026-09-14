#ifndef DEEPSTREAM_APP_LICENSE_GATE_INTERNAL_HPP_
#define DEEPSTREAM_APP_LICENSE_GATE_INTERNAL_HPP_

#include "license_gate.h"

#include <array>
#include <string>
#include <vector>

namespace ds_license
{

bool fingerprint_serial(const std::string &serial, std::string *fingerprint,
                        std::string *error);

bool find_serial(const std::vector<std::string> &paths, std::string *serial,
                 std::string *error);

DsLicenseStatus validate_for_fingerprint(
    const std::string &license_path, const std::string &fingerprint,
    const std::array<unsigned char, 32> &public_key,
    const std::string &expected_key_id, DsLicenseInfo *info);

} // namespace ds_license

#endif
