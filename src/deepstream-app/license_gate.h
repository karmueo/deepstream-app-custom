#ifndef DEEPSTREAM_APP_LICENSE_GATE_H_
#define DEEPSTREAM_APP_LICENSE_GATE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS_LICENSE_DEFAULT_PATH "/etc/deepstream-app-custom/license.lic"
#define DS_LICENSE_FINGERPRINT_SIZE 72
#define DS_LICENSE_ID_SIZE 80
#define DS_LICENSE_CUSTOMER_SIZE 256
#define DS_LICENSE_ERROR_SIZE 256

typedef enum
{
    DS_LICENSE_OK = 0,
    DS_LICENSE_DEVICE_UNAVAILABLE,
    DS_LICENSE_NOT_FOUND,
    DS_LICENSE_IO_ERROR,
    DS_LICENSE_FORMAT_ERROR,
    DS_LICENSE_KEY_ERROR,
    DS_LICENSE_SIGNATURE_ERROR,
    DS_LICENSE_PRODUCT_MISMATCH,
    DS_LICENSE_DEVICE_MISMATCH
} DsLicenseStatus;

typedef struct
{
    DsLicenseStatus status;
    char device_fingerprint[DS_LICENSE_FINGERPRINT_SIZE];
    char license_id[DS_LICENSE_ID_SIZE];
    char customer[DS_LICENSE_CUSTOMER_SIZE];
    char error[DS_LICENSE_ERROR_SIZE];
} DsLicenseInfo;

DsLicenseStatus ds_license_get_device_fingerprint(
    char fingerprint[DS_LICENSE_FINGERPRINT_SIZE],
    char error[DS_LICENSE_ERROR_SIZE]);

DsLicenseStatus ds_license_validate(const char *license_path,
                                    DsLicenseInfo *info);

DsLicenseStatus ds_license_write_request(const char *output_path,
                                         const char *app_version,
                                         DsLicenseInfo *info);

const char *ds_license_status_name(DsLicenseStatus status);

#ifdef __cplusplus
}
#endif

#endif
