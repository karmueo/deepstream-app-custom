//
// Create by Daniel Lee on 2023/9/22
//
#include <cstdlib>
#include <string>
#include <cmath>
#include "mixformer_trt.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef TIME
struct timeval tv;
uint64_t time_last;
double time_ms;
#endif

MixformerTRT::MixformerTRT(const std::string &engine_name)
{
    // deserialize engine
    this->deserialize_engine(engine_name);

    auto out_dims_0 = this->engine->getTensorShape("pred_boxes");
    // auto out_dims_0 = this->engine->getBindingDimensions(3);
    for (int j = 0; j < out_dims_0.nbDims; j++)
    {
        this->output_pred_boxes_size *= out_dims_0.d[j];
    }

    auto out_dims_1 = this->engine->getTensorShape("pred_scores");
    // auto out_dims_1 = this->engine->getBindingDimensions(4);
    for (int j = 0; j < out_dims_1.nbDims; j++)
    {
        this->output_pred_scores_size *= out_dims_1.d[j];
    }

    this->output_pred_boxes = new float[this->output_pred_boxes_size];
    this->output_pred_scores = new float[this->output_pred_scores_size];
}

MixformerTRT::~MixformerTRT()
{
    delete context;
    delete engine;
    delete runtime;
    delete[] trt_model_stream;
    delete[] this->output_pred_boxes;
    delete[] this->output_pred_scores;
    // cudaStreamDestroy(stream);
}

void MixformerTRT::deserialize_engine(const std::string &engine_name)
{
    // create a model using the API directly and serialize it to a stream
    // char *trt_model_stream{nullptr};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        this->trt_model_stream = new char[this->size];
        assert(this->trt_model_stream);
        file.read(trt_model_stream, this->size);
        file.close();
    }

    this->runtime = createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trt_model_stream,
                                                        this->size);
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();
    assert(context != nullptr);
    // delete[] trt_model_stream;
}

void MixformerTRT::infer(
    float *input_imt,
    float *input_imot,
    float *input_imsearch,
    float *output_pred_boxes,
    float *output_pred_scores,
    cv::Size input_imt_shape,
    cv::Size input_imot_shape,
    cv::Size input_imsearch_shape)
{
    assert(engine->getNbIOTensors() == 5);
    void *buffers[5];

    assert(engine->getTensorDataType(INPUT_BLOB_IMGT_NAME) == nvinfer1::DataType::kFLOAT);
    assert(engine->getTensorDataType(INPUT_BLOB_IMGOT_NAME) == nvinfer1::DataType::kFLOAT);
    assert(engine->getTensorDataType(INPUT_BLOB_IMGSEARCH_NAME) == nvinfer1::DataType::kFLOAT);

    assert(engine->getTensorDataType(OUTPUT_BLOB_PREDBOXES_NAME) == nvinfer1::DataType::kFLOAT);
    assert(engine->getTensorDataType(OUTPUT_BLOB_PREDSCORES_NAME) == nvinfer1::DataType::kFLOAT);

    // create gpu buffer on devices
    // INPUT_BLOB_IMGT_NAME: 1,3,112,112
    CHECK(cudaMalloc(&buffers[0], 3 * input_imt_shape.height * input_imt_shape.width * sizeof(float)));
    // INPUT_BLOB_IMGOT_NAME: 1,3,112,112
    CHECK(cudaMalloc(&buffers[1], 3 * input_imot_shape.height * input_imot_shape.width * sizeof(float)));
    // INPUT_BLOB_IMGSEARCH_NAME: 1,3,224,224
    CHECK(cudaMalloc(&buffers[2], 3 * input_imsearch_shape.height * input_imsearch_shape.width * sizeof(float)));
    // OUTPUT_BLOB_PREDBOXES_NAME: 1,4
    CHECK(cudaMalloc(&buffers[3], this->output_pred_boxes_size * sizeof(float)));
    // OUTPUT_BLOB_PREDSCORES_NAME: 1
    CHECK(cudaMalloc(&buffers[4], this->output_pred_scores_size * sizeof(float)));

    // create stream
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch  data to device, infer on the batch asynchronously,  and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input_imt, 3 * input_imt_shape.height * input_imt_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[1], input_imot, 3 * input_imot_shape.height * input_imot_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[2], input_imsearch, 3 * input_imsearch_shape.height * input_imsearch_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));

    context->setTensorAddress(INPUT_BLOB_IMGT_NAME, buffers[0]);
    context->setTensorAddress(INPUT_BLOB_IMGOT_NAME, buffers[1]);
    context->setTensorAddress(INPUT_BLOB_IMGSEARCH_NAME, buffers[2]);
    context->setTensorAddress(OUTPUT_BLOB_PREDBOXES_NAME, buffers[3]);
    context->setTensorAddress(OUTPUT_BLOB_PREDSCORES_NAME, buffers[4]);

    // inference
    context->enqueueV3(stream);

    CHECK(cudaMemcpyAsync(output_pred_boxes, buffers[3], this->output_pred_boxes_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_pred_scores, buffers[4], this->output_pred_scores_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release buffers
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaFree(buffers[2]));
    CHECK(cudaFree(buffers[3]));
    CHECK(cudaFree(buffers[4]));
    // std::cout << ">>>debug infer end. "  << std::endl;

    // 在函数末尾添加
    CHECK(cudaStreamDestroy(stream));
}

// put z and x into transform
void MixformerTRT::transform(cv::Mat &mat_z, cv::Mat &mat_oz, cv::Mat &mat_x)
{
    this->blob_from_image_half(mat_z, mat_oz, mat_x);
}

void MixformerTRT::blob_from_image_half(cv::Mat &img, cv::Mat &imgot, cv::Mat &imgx)
{

    cv::Mat imt_t;
    cv::Mat imot_t;
    cv::Mat imx_t;
    // cv::imshow("BGR", img);
    // cv::waitKey(500);
    cvtColor(img, imt_t, cv::COLOR_BGR2RGB);
    // cv::imshow("RGB", imt_t);
    // cv::waitKey(500);
    cvtColor(imgot, imot_t, cv::COLOR_BGR2RGB);
    cvtColor(imgx, imx_t, cv::COLOR_BGR2RGB);

    // 需及时释放
    this->input_imt = new float[img.total() * 3];       // Use __fp16 data type for blob array
    this->input_imot = new float[imgot.total() * 3];    // Use __fp16 data type for blob array
    this->input_imsearch = new float[imgx.total() * 3]; // Use __fp16 data type for blob array

    half_norm(imt_t, this->input_imt);
    half_norm(imot_t, this->input_imot);
    half_norm(imx_t, this->input_imsearch);
}

void MixformerTRT::half_norm(const cv::Mat &img, float *input_data)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    cv::Mat img_cp;
    img_cp = img.clone();

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                input_data[c * img_w * img_h + h * img_w + w] =
                    cv::saturate_cast<float>((((float)img_cp.at<cv::Vec3b>(h, w)[c]) - mean_vals[c]) * norm_vals[c]);
            }
        }
    }
}

void MixformerTRT::blob_from_image_half(cv::Mat &img, float *input_blob_half)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                input_blob_half[c * img_w * img_h + h * img_w + w] = float(img.at<cv::Vec3b>(h, w)[c]);
                // cv::saturate_cast<half_float::half>((((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std_var[c]);
                // cv::saturate_cast<half_float::half>((float)img.at<cv::Vec3b>(h, w)[c]);
                // std::cout << input_blob_half[c * img_w * img_h + h * img_w + w] << std::endl;
            }
        }
    }
}

void MixformerTRT::init(const cv::Mat &img, DrOBB bbox)
{
    // get subwindow
    cv::Mat zt_patch;
    cv::Mat ozt_patch;
    float resize_factor = 1.f;
    this->sample_target(img, zt_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    this->sample_target(img, ozt_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    // cv::Mat oz_patch = z_patch.clone();
    this->z_patch = zt_patch;
    this->oz_patch = ozt_patch;
    this->state = bbox.box;
    this->class_id = bbox.class_id;
}

const DrOBB &MixformerTRT::track(const cv::Mat &img)
{
    DrBBox pred_box = {0, 0, 0, 0};
    // if (img.empty()) return;
    // get subwindow
    cv::Mat x_patch;
    this->frame_id += 1;
    float resize_factor = 1.f;
    this->sample_target(img, x_patch, this->state, this->cfg.search_factor, this->cfg.search_size, resize_factor);

    if (x_patch.data == nullptr || this->z_patch.data == nullptr || this->oz_patch.data == nullptr)
    {
        memset(&object_box, 0, sizeof(object_box));
        return object_box;
    }
    // preprocess input tensor
    this->transform(this->z_patch, this->oz_patch, x_patch);

    // inference score， size  and offsets
    cv::Size input_imt_shape = this->z_patch.size();
    cv::Size input_imot_shape = this->oz_patch.size();
    cv::Size input_imsearch_shape = x_patch.size();

    this->infer(input_imt,
                input_imot,
                input_imsearch,
                output_pred_boxes,
                output_pred_scores,
                input_imt_shape,
                input_imot_shape,
                input_imsearch_shape);

    delete[] this->input_imt;
    delete[] this->input_imot;
    delete[] this->input_imsearch;

    float pred_score;

    this->cal_bbox(output_pred_boxes, output_pred_scores, pred_box, pred_score, resize_factor);
    if (pred_score < 0.1)
    {
        memset(&object_box, 0, sizeof(object_box));
        return object_box;
    }

    this->map_box_back(pred_box, resize_factor);
    this->clip_box(pred_box, img.rows, img.cols, 10);

    object_box.box = pred_box;
    object_box.class_id = this->class_id;
    object_box.score = pred_score;

    this->state = object_box.box;

    this->max_pred_score = this->max_pred_score * this->max_score_decay;
    // update template
    if (pred_score > 0.9 && pred_score > this->max_pred_score)
    {
        this->sample_target(img, this->max_oz_patch, this->state, this->cfg.template_factor, this->cfg.template_size, resize_factor);
        this->max_pred_score = pred_score;
    }

    if (this->frame_id % this->cfg.update_interval == 0)
    {
        this->oz_patch = this->max_oz_patch;
        this->max_pred_score = -1.0;
        this->max_oz_patch = this->oz_patch;
    }

    return object_box;
}

// calculate bbox
void MixformerTRT::cal_bbox(float *boxes_ptr, float *scores_ptr, DrBBox &pred_box, float &max_score, float resize_factor)
{
    auto cx = boxes_ptr[0];
    auto cy = boxes_ptr[1];
    auto w = boxes_ptr[2];
    auto h = boxes_ptr[3];
    if (cx < 0 || cy < 0 || w <= 0 || h <= 0)
    {
        pred_box.x0 = 0;
        pred_box.y0 = 0;
        pred_box.x1 = 0;
        pred_box.y1 = 0;
        max_score = 0;
        return;
    }

    // std::cout << "cal_bbox cx cy w h "<< cx << " " << cy << " " << w << " " << h << std::endl;
    cx = cx * this->cfg.search_size / resize_factor;
    cy = cy * this->cfg.search_size / resize_factor;
    w = w * this->cfg.search_size / resize_factor;
    h = h * this->cfg.search_size / resize_factor;

    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;

    max_score = scores_ptr[0];
}

void MixformerTRT::map_box_back(DrBBox &pred_box, float resize_factor)
{
    float cx_prev = this->state.x0 + 0.5 * (this->state.x1 - this->state.x0);
    float cy_prev = this->state.y0 + 0.5 * (this->state.y1 - this->state.y0);

    float half_side = 0.5 * this->cfg.search_size / resize_factor;

    float w = pred_box.x1 - pred_box.x0;
    float h = pred_box.y1 - pred_box.y0;
    float cx = pred_box.x0 + 0.5 * w;
    float cy = pred_box.y0 + 0.5 * h;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
}

void MixformerTRT::clip_box(DrBBox &box, int height, int wight, int margin)
{
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}

void MixformerTRT::sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor)
{
    if (target_bb.x0 < 0 || target_bb.y0 < 0 || target_bb.x1 <= target_bb.x0 || target_bb.y1 <= target_bb.y0)
    {
        std::cout << "target_bb is out of range" << std::endl;
        return;
    }

    /* Extracts a square crop centrered at target_bb box, of are search_area_factor^2 times target_bb area

    args:
        im: Img image
        target_bb - target box [x0, y0, x1, y1]
        search_area_factor - Ratio of crop size to target size
        output_sz - Size

    */
    int x = target_bb.x0;
    int y = target_bb.y0;
    int w = target_bb.x1 - target_bb.x0;
    int h = target_bb.y1 - target_bb.y0;
    int crop_sz = std::ceil(std::sqrt(w * h) * search_area_factor);

    float cx = x + 0.5 * w;
    float cy = y + 0.5 * h;
    int x1 = std::round(cx - crop_sz * 0.5);
    int y1 = std::round(cy - crop_sz * 0.5);

    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - im.cols + 1, 0);

    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2 - im.rows + 1, 0);

    // Crop target
    cv::Rect roi_rect(x1 + x1_pad, y1 + y1_pad, (x2 - x2_pad) - (x1 + x1_pad), (y2 - y2_pad) - (y1 + y1_pad));
    if (roi_rect.x < 0 || roi_rect.y < 0 || roi_rect.width <= 0 || roi_rect.height <= 0)
    {
        std::cout << "roi_rect is out of range" << std::endl;
        return;
    }
    cv::Mat roi = im(roi_rect);

    // Pad
    cv::copyMakeBorder(roi, croped, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

    // Resize
    cv::resize(croped, croped, cv::Size(output_sz, output_sz));

    resize_factor = output_sz * 1.f / crop_sz;
}
