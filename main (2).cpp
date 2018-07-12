
// ������opencvͷ�ļ�
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h> 
#include<opencv2/imgproc/imgproc.hpp>

// c++ IO��
#include<iostream>

// cv�����ռ�
using namespace cv;


/**************************************************
���ܣ���Ե����
������src-��ֵͼ��
***************************************************/
void traceBoundary(Mat src, Mat& dst)
{
	//��ʼ�߽��͵�ǰ�߽��  
	Point ptStart;
	Point ptCur;
	//������������{���£��£����£��ң����ϣ��ϣ����ϣ���}  
	int Direction[8][2] = { { -1, 1 },{ 0, 1 },{ 1, 1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 } };
	int nCurDirect = 0;//��ǰ̽��ķ���  
					   //������ʼ������ptCur==ptStart�����������һ�ֿ�ʼ��һ�ֽ�����  
	bool bAtStartPt;

	//�㷨������߽��ϵĵ㣬��ͼ�����������Ϊ��  
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
	//����ɨ��  
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
					//��һ��̽��λ��  
					xPos = ptCur.x + Direction[nCurDirect][0];
					yPos = ptCur.y + Direction[nCurDirect][1];
					int nSearchTimes = 1;
					while (src.at<uchar>(yPos, xPos) == 0)
					{
						nCurDirect++;//��ʱ����ת45��  
						if (nCurDirect >= 8)
							nCurDirect -= 8;
						xPos = ptCur.x + Direction[nCurDirect][0];
						yPos = ptCur.y + Direction[nCurDirect][1];
						//8�����ж�û�б߽�㣬˵���ǹ�����  
						if (++nSearchTimes >= 8)
						{
							xPos = ptCur.x;
							yPos = ptCur.y;
							break;
						}
					}
					//�ҵ���һ���߽��  
					ptCur.x = xPos;
					ptCur.y = yPos;
					//�������ϱ�Ǳ߽�  
					dst.at<uchar>(ptCur.y, ptCur.x) = 255;
					/***********
					�˴����Զ���vector�洢��ȡ�ı߽��
					************/
					//����ǰ̽�鷽��˳ʱ���ת90����Ϊ��һ�ε�̽���ʼ����  
					nCurDirect -= 2;
					if (nCurDirect < 0)
					{
						nCurDirect += 8;
					}
				}
				return;
			}
			//�����ڶ���߽�ʱ���ڴ˴������Ӧ���룬��ɾ��return��ÿ������һ���߽磬ɾ����Ӧ������  
		}
	}
}


// main����
int main()
{
	char *tttt;
	char *aaaa = "��ϸ��";
	char *bbbb = "���ǰ�ϸ��";
	printf("\t***************************\n");
	printf("\t****  ��ϸ��ʶ��ϵͳ  *****\n");
	printf("\t***************************\n");
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    //��ȡͼ��Դ
    cv::Mat srcImage = cv::imread("demo.jpg");  // ͼƬ·��
	//��СͼƬ�ĳߴ�
	resize(srcImage, srcImage, Size(srcImage.cols / 2, srcImage.rows / 2), 0, 0, INTER_LINEAR);
	//���ͼƬ��ʧ�����˳�
    if (srcImage.empty()) 
	{
        return -1;  // �˳�
    }
	printf("ͼ������ɹ�������\n\n");
    //��һ����תΪ�Ҷ�ͼ��
    cv::Mat srcGray;//�����޳�ʼ������
    //cv::cvtColor(srcImage, srcGray,CV_RGB2GRAY);// ͼ��ĻҶ�ת��

	// HSV�ķ�����Ч�����ѣ�
	Mat img_h, img_s, img_v, imghsv;
	vector<cv::Mat> hsv_vec;  // ͼ������
	cvtColor(srcImage, imghsv, CV_BGR2HSV);  // hsvͨ����ת��
	split(imghsv, hsv_vec);   //�ֿ�
	img_h = hsv_vec[0]; // Hͨ��
	img_s = hsv_vec[1]; // Sͨ��
	img_v = hsv_vec[2]; // Vͨ��

	// ��ȡRGB�е�Rͨ�����з�����Ч����ã�
	Mat bgr(srcImage.rows, srcImage.cols, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 2; i < 3; i++)
	{
		Mat temp(srcImage.rows, srcImage.cols, CV_8UC1); // ��ͨ����ͼ��
		Mat out[] = { bgr };
		int from_to[] = { i,i };
		mixChannels(&srcImage, 1, out, 1, from_to, 1);  // ����ͨ������ȡ
		cv::cvtColor(bgr, bgr, CV_RGB2GRAY);// ͼ��ĻҶ�ת��
		//�������һ��ͨ�������ݽ��з���  
		//if (i == 0) imshow("��ͨ��ͼ����ɫͨ����", bgr);
		//if (i == 1) imshow("��ͨ��ͼ����ɫͨ����", bgr);
		//if (i == 2) imshow("��ͨ��ͼ�񣨺�ɫͨ����", bgr);
		//waitKey();
	}
	// Rͨ����Ϊͼ�������
	srcGray = bgr;

	Mat srcGray_output = srcGray.clone();   // ����
	printf("��һ����ͼ��ҶȻ�����ɹ�...\n");
	// �ڶ�������ֵ��
	threshold(srcGray, srcGray, 15, 255, CV_THRESH_BINARY);  //ͼ���ֵ��
	Mat erzhihua_IMAGE = srcGray.clone();  
	printf("�ڶ�����ͼ���ֵ������ɹ�...\n");
	//����������̬ѧ�Ĵ���
	//������ (ȥ��һЩ���)  �����ֵ����ͼƬ���Ų�����Ȼ�ܶ࣬���������size  
	Mat element_OPEN = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat element_CLOSE = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element_close00 = getStructuringElement(MORPH_RECT, Size(8, 8));
	morphologyEx(srcGray, srcGray, MORPH_OPEN, element_OPEN);
	morphologyEx(srcGray, srcGray, MORPH_CLOSE, element_CLOSE);
	Mat close00;
	morphologyEx(srcGray, close00, MORPH_CLOSE, element_close00);
	printf("��������ͼ����̬ѧת����ɹ�...\n");
	//���Ĳ�����������
	Mat noise_IMAGE = srcGray.clone();
	for (int x = 0; x<noise_IMAGE.rows; x++)
	{
		for (int y = 0; y<noise_IMAGE.cols; y++)
		{
			if (x > 180 && y < 140)   noise_IMAGE.at<uchar>(x, y) = 0;  // ȥ������
		}
	}
	printf("���Ĳ���ͼ��ȥ��������ɹ�...\n");
	// ���岽����Ե������
	Mat OUTPUT_IMAGE = srcGray_output.clone();
	Mat OUTPUT_SOBEL_x = srcGray_output.clone();
	Mat OUTPUT_SOBEL_y = srcGray_output.clone();
	Mat OUTPUT_SOBEL = srcGray_output.clone();
	//Sobel��Ե����
	// X�����ı�Ե����
	Sobel(srcGray_output, OUTPUT_SOBEL_x, srcGray_output.depth(), 1, 0, 1, 1, 0, BORDER_DEFAULT);
	// Y�����ı�Ե����
	Sobel(srcGray_output, OUTPUT_SOBEL_y, srcGray_output.depth(), 0, 1, 1, 1, 0, BORDER_DEFAULT);
	float alpha = 10;  // x�� ����ϵ��
	float beta = 10;   // y�� ����ϵ��
	addWeighted(OUTPUT_SOBEL_x, alpha, OUTPUT_SOBEL_y, beta, 0.0, OUTPUT_SOBEL);
	printf("���岽��ͼ���Ե���ٴ���ɹ�...\n");
	// ����������̬ѧ�����������Բ�ȡ����ζȡ��쳤�ȣ�
	long SUM_mianji = 0;  // ������
	long SUM_yuandu = 0;  // Բ�ȼ��
	long SUM_juxing = 0;  // ���ζȼ��
	long SUM_shenchang = 0;  // �쳤�ȼ��

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
	// ����صĹ�һ��
	SUM_mianji /= noise_IMAGE.rows;  
	SUM_yuandu /= noise_IMAGE.rows;
	SUM_juxing /= noise_IMAGE.rows;
	SUM_shenchang /= noise_IMAGE.rows;

	// �����б�
	if(SUM_mianji > 400 && SUM_yuandu > 150 && SUM_juxing > 400 && SUM_shenchang > 200 )
		tttt = aaaa;
	else
		tttt = bbbb;

	printf("��������ͼ�����ʶ��ɹ�...\n\n");
	printf("ʶ�����ݣ���%s��\n",tttt);
    cv::imshow("ԭͼ��", srcImage);//��ʾԴͼ��
    cv::imshow("�ҶȻ�", srcGray_output);//��ʾ�Ҷ�ͼ��
	cv::imshow("��ֵ��", erzhihua_IMAGE);//��ʾ��ֵ��
	cv::imshow("��̬ѧ", srcGray);//��ʾ��̬ѧ
	cv::imshow("ȥ����", noise_IMAGE);//��ʾȥ��
	cv::imshow("�߽����", OUTPUT_SOBEL);//��ʾ�߽����
    cv::waitKey(0); //�ȴ���������ͼƬһֱ��ʾ�⡣
    return 0;
}