
// 包含的opencv头文件
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h> 
#include<opencv2/imgproc/imgproc.hpp>

// c++ IO流
#include<iostream>

// cv命名空间
using namespace cv;


/**************************************************
功能：边缘跟踪
参数：src-二值图像
***************************************************/
void traceBoundary(Mat src, Mat& dst)
{
	//起始边界点和当前边界点  
	Point ptStart;
	Point ptCur;
	//搜索方向数组{左下，下，右下，右，右上，上，左上，左}  
	int Direction[8][2] = { { -1, 1 },{ 0, 1 },{ 1, 1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 } };
	int nCurDirect = 0;//当前探查的方向  
					   //搜索开始，区别ptCur==ptStart的两种情况（一种开始，一种结束）  
	bool bAtStartPt;

	//算法不处理边界上的点，将图像的四周设置为白  
	//for (int i = 0; i < src.rows; i++)  
	//{  
	//  dst.at<uchar>(i, 0) = 255;  
	//  dst.at<uchar>(i, src.rows - 1) = 255;  
	//}  
	//for (int j = 0; j < src.cols; j++)  
	//{  
	//  dst.at<uchar>(0, j) = 255;  
	//  dst.at<uchar>(src.rows - 1, j) = 255;  
	//}  

	int xPos, yPos;
	//逐行扫描  
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) > 0)
			{
				ptStart.x = j;
				ptStart.y = i;

				ptCur = ptStart;
				bAtStartPt = true;
				while ((ptCur.x != ptStart.x) || (ptCur.y != ptStart.y) || bAtStartPt)
				{
					bAtStartPt = false;
					//下一个探查位置  
					xPos = ptCur.x + Direction[nCurDirect][0];
					yPos = ptCur.y + Direction[nCurDirect][1];
					int nSearchTimes = 1;
					while (src.at<uchar>(yPos, xPos) == 0)
					{
						nCurDirect++;//逆时针旋转45度  
						if (nCurDirect >= 8)
							nCurDirect -= 8;
						xPos = ptCur.x + Direction[nCurDirect][0];
						yPos = ptCur.y + Direction[nCurDirect][1];
						//8领域中都没有边界点，说明是孤立点  
						if (++nSearchTimes >= 8)
						{
							xPos = ptCur.x;
							yPos = ptCur.y;
							break;
						}
					}
					//找到下一个边界点  
					ptCur.x = xPos;
					ptCur.y = yPos;
					//在新像上标记边界  
					dst.at<uchar>(ptCur.y, ptCur.x) = 255;
					/***********
					此处可以定义vector存储提取的边界点
					************/
					//将当前探查方向顺时针回转90度作为下一次的探查初始方向  
					nCurDirect -= 2;
					if (nCurDirect < 0)
					{
						nCurDirect += 8;
					}
				}
				return;
			}
			//当存在多个边界时，在此处添加相应代码，并删除return（每跟踪完一个边界，删除相应的区域）  
		}
	}
}


// main函数
int main()
{
	char *tttt;
	char *aaaa = "癌细胞";
	char *bbbb = "不是癌细胞";
	printf("\t***************************\n");
	printf("\t****  癌细胞识别系统  *****\n");
	printf("\t***************************\n");
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    //读取图像源
    cv::Mat srcImage = cv::imread("demo.jpg");  // 图片路径
	//缩小图片的尺寸
	resize(srcImage, srcImage, Size(srcImage.cols / 2, srcImage.rows / 2), 0, 0, INTER_LINEAR);
	//如果图片打开失败则退出
    if (srcImage.empty()) 
	{
        return -1;  // 退出
    }
	printf("图像输入成功！！！\n\n");
    //第一步：转为灰度图像
    cv::Mat srcGray;//创建无初始化矩阵
    //cv::cvtColor(srcImage, srcGray,CV_RGB2GRAY);// 图像的灰度转换

	// HSV的方法（效果不佳）
	Mat img_h, img_s, img_v, imghsv;
	vector<cv::Mat> hsv_vec;  // 图像容器
	cvtColor(srcImage, imghsv, CV_BGR2HSV);  // hsv通道的转换
	split(imghsv, hsv_vec);   //分开
	img_h = hsv_vec[0]; // H通道
	img_s = hsv_vec[1]; // S通道
	img_v = hsv_vec[2]; // V通道

	// 获取RGB中的R通道进行分析（效果最好）
	Mat bgr(srcImage.rows, srcImage.cols, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 2; i < 3; i++)
	{
		Mat temp(srcImage.rows, srcImage.cols, CV_8UC1); // 单通道的图像
		Mat out[] = { bgr };
		int from_to[] = { i,i };
		mixChannels(&srcImage, 1, out, 1, from_to, 1);  // 单个通道的提取
		cv::cvtColor(bgr, bgr, CV_RGB2GRAY);// 图像的灰度转换
		//获得其中一个通道的数据进行分析  
		//if (i == 0) imshow("单通道图像（蓝色通道）", bgr);
		//if (i == 1) imshow("单通道图像（绿色通道）", bgr);
		//if (i == 2) imshow("单通道图像（红色通道）", bgr);
		//waitKey();
	}
	// R通道作为图像的输入
	srcGray = bgr;

	Mat srcGray_output = srcGray.clone();   // 拷贝
	printf("第一步：图像灰度化处理成功...\n");
	// 第二步：二值化
	threshold(srcGray, srcGray, 15, 255, CV_THRESH_BINARY);  //图像二值化
	Mat erzhihua_IMAGE = srcGray.clone();  
	printf("第二步：图像二值化处理成功...\n");
	//第三步：形态学的处理
	//开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size  
	Mat element_OPEN = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat element_CLOSE = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element_close00 = getStructuringElement(MORPH_RECT, Size(8, 8));
	morphologyEx(srcGray, srcGray, MORPH_OPEN, element_OPEN);
	morphologyEx(srcGray, srcGray, MORPH_CLOSE, element_CLOSE);
	Mat close00;
	morphologyEx(srcGray, close00, MORPH_CLOSE, element_close00);
	printf("第三步：图像形态学转处理成功...\n");
	//第四步：噪声处理
	Mat noise_IMAGE = srcGray.clone();
	for (int x = 0; x<noise_IMAGE.rows; x++)
	{
		for (int y = 0; y<noise_IMAGE.cols; y++)
		{
			if (x > 180 && y < 140)   noise_IMAGE.at<uchar>(x, y) = 0;  // 去除噪声
		}
	}
	printf("第四步：图像去噪声处理成功...\n");
	// 第五步：边缘检测跟踪
	Mat OUTPUT_IMAGE = srcGray_output.clone();
	Mat OUTPUT_SOBEL_x = srcGray_output.clone();
	Mat OUTPUT_SOBEL_y = srcGray_output.clone();
	Mat OUTPUT_SOBEL = srcGray_output.clone();
	//Sobel边缘处理
	// X分量的边缘处理
	Sobel(srcGray_output, OUTPUT_SOBEL_x, srcGray_output.depth(), 1, 0, 1, 1, 0, BORDER_DEFAULT);
	// Y分量的边缘处理
	Sobel(srcGray_output, OUTPUT_SOBEL_y, srcGray_output.depth(), 0, 1, 1, 1, 0, BORDER_DEFAULT);
	float alpha = 10;  // x的 叠加系数
	float beta = 10;   // y的 叠加系数
	addWeighted(OUTPUT_SOBEL_x, alpha, OUTPUT_SOBEL_y, beta, 0.0, OUTPUT_SOBEL);
	printf("第五步：图像边缘跟踪处理成功...\n");
	// 第六步：形态学特征（面积、圆度、矩形度、伸长度）
	long SUM_mianji = 0;  // 面积检测
	long SUM_yuandu = 0;  // 圆度检测
	long SUM_juxing = 0;  // 矩形度检测
	long SUM_shenchang = 0;  // 伸长度检测

	for (int x = 0; x<noise_IMAGE.rows; x++)
	{
		for (int y = 0; y<noise_IMAGE.cols; y++)
		{
			if (noise_IMAGE.at<uchar>(x, y) > 150)   SUM_mianji++;
			if (noise_IMAGE.at<uchar>(x, y) < 50)   SUM_yuandu++;
			if (erzhihua_IMAGE.at<uchar>(x, y) > 15)   SUM_juxing++;
			if (OUTPUT_SOBEL.at<uchar>(x, y) > 10)   SUM_shenchang++;
		}
	}
	// 行相关的归一化
	SUM_mianji /= noise_IMAGE.rows;  
	SUM_yuandu /= noise_IMAGE.rows;
	SUM_juxing /= noise_IMAGE.rows;
	SUM_shenchang /= noise_IMAGE.rows;

	// 进行判别
	if(SUM_mianji > 400 && SUM_yuandu > 150 && SUM_juxing > 400 && SUM_shenchang > 200 )
		tttt = aaaa;
	else
		tttt = bbbb;

	printf("第六步：图像分类识别成功...\n\n");
	printf("识别内容：【%s】\n",tttt);
    cv::imshow("原图像", srcImage);//显示源图像
    cv::imshow("灰度化", srcGray_output);//显示灰度图像
	cv::imshow("二值化", erzhihua_IMAGE);//显示二值化
	cv::imshow("形态学", srcGray);//显示形态学
	cv::imshow("去噪声", noise_IMAGE);//显示去燥
	cv::imshow("边界跟踪", OUTPUT_SOBEL);//显示边界跟踪
    cv::waitKey(0); //等待。可以让图片一直显示这。
    return 0;
}