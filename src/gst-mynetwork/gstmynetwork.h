#ifndef __GST__MYNETWORK_H__
#define __GST__MYNETWORK_H__

#include <gst/gst.h>

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty IP integration plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

#define GST_TYPE__MYNETWORK (gst_mynetwork_get_type())
G_DECLARE_FINAL_TYPE(Gstmynetwork, gst_mynetwork,
                     GST, _MYNETWORK, GstElement)

struct _Gstmynetwork
{
    GstElement element;

    GstPad *sinkpad, *srcpad;

    gboolean silent;
};

G_END_DECLS

#endif /* __GST__MYNETWORK_H__ */
