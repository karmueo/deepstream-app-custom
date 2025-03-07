#ifndef __GST__MYNETWORK_H__
#define __GST__MYNETWORK_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#define PACKAGE "_mynetwork"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "My plugin for Deepstream Network"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty IP integration plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstmynetworkClass GstmynetworkClass;
typedef struct _Gstmynetwork Gstmynetwork;

#define GST_TYPE_MYNETWORK (gst_mynetwork_get_type())
#define GST_MYNETWORK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MYNETWORK, Gstmynetwork))

// G_DECLARE_FINAL_TYPE(Gstmynetwork,
//                      gst_mynetwork,
//                      GST, _MYNETWORK,
//                      GstElement)

struct _Gstmynetwork
{
    GstBaseSink parent;

    guint unique_id;

    guint gpu_id;

    gboolean silent;
};

struct _GstmynetworkClass
{
    GstBaseSinkClass parent_class;
};

GType gst_mynetwork_get_type(void);

G_END_DECLS

#endif /* __GST__MYNETWORK_H__ */
