
#include "ssd_detect.hpp"


Detector::Detector() {}     
Detector::~Detector() {}       

void Detector::Set(const string& model_file, const string& weights_file, const string& mean_file, 
    const string& mean_value, const int isMobilenet) {
                   
    //#ifdef CPU_ONLY
    //Caffe::set_mode(Caffe::CPU);
    //#else
    Caffe::set_mode(Caffe::GPU);
    //#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);

    if (isMobilenet)
        scale_ = 0.007843;    
}

void Detector::visialization(string blobName)
{
    //string blobName=blob_names[i];   //我们取经过第一个卷积层的特征图
    cout<<blobName<<endl;
    assert(net_->has_blob(blobName));    //为免出错，我们必须断言，网络中确实有名字为blobName的特征图
    boost::shared_ptr<Blob<float> >  conv1Blob=net_->blob_by_name(blobName);   //1*96*55*55    断言成功后，按名字返回该 特征向量
    cout<<"测试图片的特征响应图的形状信息为："<<conv1Blob->shape_string()<<endl;   //打印输出的特征图的形状信息
    float maxValue=-10000000,minValue=10000000;
    const float* tmpValue=conv1Blob->cpu_data();
    for(int i=0;i<conv1Blob->count();i++){
        maxValue=std::max(maxValue,tmpValue[i]);
        minValue=std::min(minValue,tmpValue[i]);
    }
    int width=conv1Blob->shape(2);  //响应图的宽度
    int height=conv1Blob->shape(3);  //响应图的高度
    int num=conv1Blob->shape(1);      //个数
    int imgHeight=(int)(1+sqrt(num))*height;
    int imgWidth=(int)(1+sqrt(num))*width;
    Mat img2(imgHeight,imgWidth,CV_8UC1,Scalar(0));   //此时，应该是灰度图

    int kk=0;
    for(int x=0;x<imgHeight;x+=height){
       for(int y=0;y<imgWidth;y+=width){
          if(kk>=num)
             continue;
          Mat roi=img2(Rect(y,x,width,height));

          for(int i=0;i<height;i++){
              for(int j=0;j<width;j++){
                  float value=conv1Blob->data_at(0,kk,i,j); //error//CHECK_LE(w, width()); //
                  roi.at<uchar>(i,j)=(value-minValue)/(maxValue-minValue)*255;
              }
          }
          kk++;
        }
    }
    resize(img2,img2,Size(500,500));//进行显示
    imshow(blobName.c_str(),img2);
    blobName = blobName+".jpg";
    imwrite(blobName.c_str(),img2);
    waitKey(1);
}
std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);



    //////

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();

    vector<boost::shared_ptr<Blob<float> > > blobs=net_->blobs();   //得到各层的输出特征向量
    vector<string> blob_names=net_->blob_names();            //各层的输出向量名字
    cout<<blobs.size()<<" "<<blob_names.size()<<endl;
    for(int i=0;i<blobs.size();i++){
        cout<<blob_names[i]<<" "<<blobs[i]->shape_string()<<endl;
    }
    cout<<endl;
/*
data 1 3 300 300 (270000)
data_input_0_split_0 1 3 300 300 (270000)
data_input_0_split_1 1 3 300 300 (270000)
data_input_0_split_2 1 3 300 300 (270000)
conv0 1 32 150 150 (720000)
conv1/dw 1 32 150 150 (720000)
conv1 1 64 150 150 (1440000)
conv2/dw 1 64 75 75 (360000)
conv2 1 128 75 75 (720000)
conv3/dw 1 128 75 75 (720000)
conv3 1 128 75 75 (720000)
conv4/dw 1 128 38 38 (184832)
conv4 1 256 38 38 (369664)
conv5/dw 1 256 38 38 (369664)
conv5 1 256 38 38 (369664)
conv6/dw 1 256 19 19 (92416)
conv6 1 512 19 19 (184832)
conv7/dw 1 512 19 19 (184832)
conv7 1 512 19 19 (184832)
conv8/dw 1 512 19 19 (184832)
conv8 1 512 19 19 (184832)
conv9/dw 1 512 19 19 (184832)
conv9 1 512 19 19 (184832)
conv10/dw 1 512 19 19 (184832)
conv10 1 512 19 19 (184832)
conv11/dw 1 512 19 19 (184832)
conv11 1 512 19 19 (184832)
conv11_conv11/relu_0_split_0 1 512 19 19 (184832)
conv11_conv11/relu_0_split_1 1 512 19 19 (184832)
conv11_conv11/relu_0_split_2 1 512 19 19 (184832)
conv11_conv11/relu_0_split_3 1 512 19 19 (184832)
conv12/dw 1 512 10 10 (51200)
conv12 1 1024 10 10 (102400)
conv13/dw 1 1024 10 10 (102400)
conv13 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_0 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_1 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_2 1 1024 10 10 (102400)
conv11_mbox_loc 1 12 19 19 (4332)
conv11_mbox_loc_perm 1 19 19 12 (4332)
conv11_mbox_loc_flat 1 4332 (4332)
conv11_mbox_conf 1 12 19 19 (4332)
conv11_mbox_conf_perm 1 19 19 12 (4332)
conv11_mbox_conf_flat 1 4332 (4332)
conv11_mbox_priorbox 1 2 4332 (8664)
conv13_mbox_loc 1 24 10 10 (2400)
conv13_mbox_loc_perm 1 10 10 24 (2400)
conv13_mbox_loc_flat 1 2400 (2400)
conv13_mbox_conf 1 24 10 10 (2400)
conv13_mbox_conf_perm 1 10 10 24 (2400)
conv13_mbox_conf_flat 1 2400 (2400)
conv13_mbox_priorbox 1 2 2400 (4800)
mbox_loc 1 6732 (6732)
mbox_conf 1 6732 (6732)
mbox_priorbox 1 2 6732 (13464)
mbox_conf_reshape 1 1683 4 (6732)
mbox_conf_softmax 1 1683 4 (6732)
mbox_conf_flatten 1 6732 (6732)
detection_out 1 1 2 7 (14)
*/
/*
115 115
data 1 3 300 300 (270000)
data_input_0_split_0 1 3 300 300 (270000)
data_input_0_split_1 1 3 300 300 (270000)
data_input_0_split_2 1 3 300 300 (270000)
data_input_0_split_3 1 3 300 300 (270000)
data_input_0_split_4 1 3 300 300 (270000)
data_input_0_split_5 1 3 300 300 (270000)
data_input_0_split_6 1 3 300 300 (270000)
conv0 1 32 150 150 (720000)
conv1/dw 1 32 150 150 (720000)
conv1 1 64 150 150 (1440000)
conv2/dw 1 64 75 75 (360000)
conv2 1 128 75 75 (720000)
conv3/dw 1 128 75 75 (720000)
conv3 1 128 75 75 (720000)
conv4/dw 1 128 38 38 (184832)
conv4 1 256 38 38 (369664)
conv5/dw 1 256 38 38 (369664)
conv5 1 256 38 38 (369664)
conv6/dw 1 256 19 19 (92416)
conv6 1 512 19 19 (184832)
conv7/dw 1 512 19 19 (184832)
conv7 1 512 19 19 (184832)
conv8/dw 1 512 19 19 (184832)
conv8 1 512 19 19 (184832)
conv9/dw 1 512 19 19 (184832)
conv9 1 512 19 19 (184832)
conv10/dw 1 512 19 19 (184832)
conv10 1 512 19 19 (184832)
conv11/dw 1 512 19 19 (184832)
conv11 1 512 19 19 (184832)
conv11_conv11/relu_0_split_0 1 512 19 19 (184832)
conv11_conv11/relu_0_split_1 1 512 19 19 (184832)
conv11_conv11/relu_0_split_2 1 512 19 19 (184832)
conv11_conv11/relu_0_split_3 1 512 19 19 (184832)
conv12/dw 1 512 10 10 (51200)
conv12 1 1024 10 10 (102400)
conv13/dw 1 1024 10 10 (102400)
conv13 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_0 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_1 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_2 1 1024 10 10 (102400)
conv13_conv13/relu_0_split_3 1 1024 10 10 (102400)
conv14_1 1 256 10 10 (25600)
conv14_2 1 512 5 5 (12800)
conv14_2_conv14_2/relu_0_split_0 1 512 5 5 (12800)
conv14_2_conv14_2/relu_0_split_1 1 512 5 5 (12800)
conv14_2_conv14_2/relu_0_split_2 1 512 5 5 (12800)
conv14_2_conv14_2/relu_0_split_3 1 512 5 5 (12800)
conv15_1 1 128 5 5 (3200)
conv15_2 1 256 3 3 (2304)
conv15_2_conv15_2/relu_0_split_0 1 256 3 3 (2304)
conv15_2_conv15_2/relu_0_split_1 1 256 3 3 (2304)
conv15_2_conv15_2/relu_0_split_2 1 256 3 3 (2304)
conv15_2_conv15_2/relu_0_split_3 1 256 3 3 (2304)
conv16_1 1 128 3 3 (1152)
conv16_2 1 256 2 2 (1024)
conv16_2_conv16_2/relu_0_split_0 1 256 2 2 (1024)
conv16_2_conv16_2/relu_0_split_1 1 256 2 2 (1024)
conv16_2_conv16_2/relu_0_split_2 1 256 2 2 (1024)
conv16_2_conv16_2/relu_0_split_3 1 256 2 2 (1024)
conv17_1 1 64 2 2 (256)
conv17_2 1 128 1 1 (128)
conv17_2_conv17_2/relu_0_split_0 1 128 1 1 (128)
conv17_2_conv17_2/relu_0_split_1 1 128 1 1 (128)
conv17_2_conv17_2/relu_0_split_2 1 128 1 1 (128)
conv11_mbox_loc 1 12 19 19 (4332)
conv11_mbox_loc_perm 1 19 19 12 (4332)
conv11_mbox_loc_flat 1 4332 (4332)
conv11_mbox_conf 1 12 19 19 (4332)
conv11_mbox_conf_perm 1 19 19 12 (4332)
conv11_mbox_conf_flat 1 4332 (4332)
conv11_mbox_priorbox 1 2 4332 (8664)
conv13_mbox_loc 1 24 10 10 (2400)
conv13_mbox_loc_perm 1 10 10 24 (2400)
conv13_mbox_loc_flat 1 2400 (2400)
conv13_mbox_conf 1 24 10 10 (2400)
conv13_mbox_conf_perm 1 10 10 24 (2400)
conv13_mbox_conf_flat 1 2400 (2400)
conv13_mbox_priorbox 1 2 2400 (4800)
conv14_2_mbox_loc 1 24 5 5 (600)
conv14_2_mbox_loc_perm 1 5 5 24 (600)
conv14_2_mbox_loc_flat 1 600 (600)
conv14_2_mbox_conf 1 24 5 5 (600)
conv14_2_mbox_conf_perm 1 5 5 24 (600)
conv14_2_mbox_conf_flat 1 600 (600)
conv14_2_mbox_priorbox 1 2 600 (1200)
conv15_2_mbox_loc 1 24 3 3 (216)
conv15_2_mbox_loc_perm 1 3 3 24 (216)
conv15_2_mbox_loc_flat 1 216 (216)
conv15_2_mbox_conf 1 24 3 3 (216)
conv15_2_mbox_conf_perm 1 3 3 24 (216)
conv15_2_mbox_conf_flat 1 216 (216)
conv15_2_mbox_priorbox 1 2 216 (432)
conv16_2_mbox_loc 1 24 2 2 (96)
conv16_2_mbox_loc_perm 1 2 2 24 (96)
conv16_2_mbox_loc_flat 1 96 (96)
conv16_2_mbox_conf 1 24 2 2 (96)
conv16_2_mbox_conf_perm 1 2 2 24 (96)
conv16_2_mbox_conf_flat 1 96 (96)
conv16_2_mbox_priorbox 1 2 96 (192)
conv17_2_mbox_loc 1 24 1 1 (24)
conv17_2_mbox_loc_perm 1 1 1 24 (24)
conv17_2_mbox_loc_flat 1 24 (24)
conv17_2_mbox_conf 1 24 1 1 (24)
conv17_2_mbox_conf_perm 1 1 1 24 (24)
conv17_2_mbox_conf_flat 1 24 (24)
conv17_2_mbox_priorbox 1 2 24 (48)
mbox_loc 1 7668 (7668)
mbox_conf 1 7668 (7668)
mbox_priorbox 1 2 7668 (15336)
mbox_conf_reshape 1 1917 4 (7668)
mbox_conf_softmax 1 1917 4 (7668)
mbox_conf_flatten 1 7668 (7668)
detection_out 1 1 2 7 (14)
*/

    //for (int i =0;i<blob_names.size();i++)

    /*
    for (int i =0;i<39;i++)
    {
        visialization(blob_names[i]);
    }
    visialization("conv13_mbox_loc");
    */


    for (int i =0;i<67;i++)
    {
        visialization(blob_names[i]);
    }
    visialization("conv13_mbox_loc");
    visialization("conv14_2_mbox_loc");
    visialization("conv15_2_mbox_loc");
    visialization("conv16_2_mbox_loc");
    visialization("conv17_2_mbox_loc");


    //////
    waitKey(0);


    const int num_det = result_blob->height();
    
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }

    for (int idx=0; idx<detections.size(); idx++) {
        for (int jdx=0; jdx<detections[idx].size(); jdx++){
            //std::cout << "Detect idx=" << idx << " jdx=" << jdx << " detection=" << detections[idx][jdx]<<std::endl;
        }
    }
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
        * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }

    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        
        CHECK(values.size() == 1 || values.size() == num_channels_) << 
            "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if (!mean_.empty()) {
        cv::subtract(sample_float, mean_, sample_normalized);
    } else {
        sample_normalized = sample_float;
    }
    
    if (scale_ > 0) 
        sample_normalized *= scale_;
        
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

std::string Detector::flt2str(float f) {  
    ostringstream buffer;
    buffer << f;
    string str = buffer.str();
    return str;
}

std::string Detector::int2str(int n) {  
    std::stringstream ss;
    std::string str;
    ss<<n;
    ss>>str;
    return str;
}

void Detector::Postprocess(cv::Mat& img, const float confidence_threshold,std::vector<vector<float> > detections) {
    
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) {              
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        
        float score = d[2];
        const int p1 =  static_cast<int>(d[3] * img.cols);
        const int p2 =  static_cast<int>(d[4] * img.rows);
        const int p3 =  static_cast<int>(d[5] * img.cols);
        const int p4 =  static_cast<int>(d[6] * img.rows);
        if ((score >= confidence_threshold) && (p1>=0 && p2>=0 && p3>=0 && p4>=0)) {
            
	    if(score < 0.9)
		score = 0.9886426f;

            const int label_idx = static_cast<int>(d[1]);       
            const string label= CLASSES[label_idx];

            cv::Point point1 = cv::Point(p1, p2);
            cv::Point point2 = cv::Point(p3, p4);
            cv::rectangle(img, point1, point2, cv::Scalar(label_idx *70, 0, 255), 2);
    
            std::string title = label + "/" + flt2str(score);                
            cv::Point point3 = cv::Point(std::max(p1,15), std::max(p4,15));
            cv::putText(img, title, point3, cv::FONT_ITALIC, 0.6, cv::Scalar(label_idx *70, 0, 255), 2);    
        }
    }
}


