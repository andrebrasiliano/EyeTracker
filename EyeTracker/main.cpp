#include <Windows.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <time.h>
#include <math.h>
#include "Header Files\tracker.h"
#include "kcftracker.hpp"
#include <stdlib.h>
#include <stdint.h>
#include "Header Files\stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "Header Files/flandmark_detector.h"
#include "Header Files/detectar.h"
#include "Header Files/centroOlho.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define MY_BUFSIZE 1024

//Variáveis
CascadeClassifier cascade, nestedCascade, pairCascade;
string cascadeName, nestedCascadeName, pairCascadeName, inputName;
VideoCapture capture;
time_t start, end;

RECT cursor;

int flag = 0, flag2 = 0, num_frames = 120, frames = 0, seg = 0, novo_seg = 0, camera = 0, radius, flagCorners = 0, countTime = 0;;
float radiusHought;
double scale = 1, fps;

bool initialized = true;
bool tryflip;

Mat kernelBinary = (Mat_<char>(3, 3) << -2, -3, -2,
	-3, 21, -3,
	-2, -3, -2);


Mat gray, cannyImg, gaussImg, image_prev, image_next, frame, image, imageCab;

vector<Point2f> eyesCorners, eyecornersNew, centersEyesNew;
vector<Point2f> corners;
vector<uchar> founded, foundedCorners;
vector<Point> features_prev, features_next;
vector<float> fail, failCorners;
vector<Vec3f> circles;
vector<Rect> nestedObjects;
vector<vector<Point>> calibrateEyes;
vector<Point> calibrateEL;
vector<Point> calibrateER;

Point pupilaEsquerda, pupilaDireita, deltas, gaze, gazeA;

Rect nr, r, tamanho_olho1, tamanho_olho2, rect, result;

//Métodos
Rect detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip);
void detectEyesCorners(Mat img);
void Frames_seg();
void setCursorPosition(int deltaXcursor, int deltaYcursor);
void calibrates(Mat frame);
void calcDeltaX_DeltaY();
void calibrates(Mat frame);
void calcNewGaze();


int main(int argc, const char** argv)
{
	Rect olhoEsquerdo, olhoDireito, origem;
	vector<Rect> olho;
	KCFTracker tracker(false, true, false, false);
	int seg = 0, novo_seg = 0, xOlhoEsquerdo, xOlhoDireito;
	vector<Point2f> centersEyes;
	Mat image_prev, image_next;

	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{cascade|../data/haarcascade_frontalface_alt.xml|}"
		"{nested-cascade|../data/haarcascade_eye_tree_eyeglasses.xml|}"
		"{pair-cascade|../data/pairCascade.xml|}"
		"{scale|1|}{try-flip||}{@filename||}"
	);

	pairCascadeName = parser.get<string>("pair-cascade");
	cascadeName = parser.get<string>("cascade");
	nestedCascadeName = parser.get<string>("nested-cascade");
	scale = parser.get<double>("scale");

	tryflip = parser.has("try-flip");
	nestedCascade.load(nestedCascadeName);
	cascade.load(cascadeName);
	pairCascade.load(pairCascadeName);


	capture.open(camera);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	image = imread(inputName, 1);


	if (capture.isOpened())
	{
		for (;;)
		{
			Frames_seg();

			capture >> frame;
			//frame = imread("../data/lena.jpg");
			if (frame.empty()) { break; }



			if (flag == 0) {
				detectAndDraw(frame, cascade, pairCascade, scale, tryflip);
				detectEyesCorners(frame);
				Rect rect = detectar::Face(frame, cascade, nestedCascade, scale, tryflip);

				if (rect.area() == 0) {
					flag = 0;
				}

				else {
					olho = detectar::OlhoProporcao(rect);
					tracker.init(rect, frame);
					flag = 1;
				}

				if (olho.size() == 0) {
					flag = 0;
				}

				else {

					origem = olho[0];

					olhoEsquerdo = olho[1];
					olhoDireito = olho[2];

					xOlhoEsquerdo = olho[1].x - olho[0].x;
					xOlhoDireito = olho[2].x - olho[0].x;

					olho[0].x = olho[0].x + rect.x;
					olho[0].y = olho[0].y + rect.y;



					tracker.init(olho[0], frame);
					flag = 1;
				}
			}

			else {

				result = tracker.update(frame);

				olhoEsquerdo = Rect(xOlhoEsquerdo + result.x, result.y, olhoEsquerdo.width, result.height);
				olhoDireito = Rect(xOlhoDireito + result.x, result.y, olhoDireito.width, result.height);

				pupilaEsquerda = centroOlho::encontrarCentro(frame, result, xOlhoEsquerdo, olhoEsquerdo.width);
				pupilaDireita = centroOlho::encontrarCentro(frame, result, xOlhoDireito, olhoDireito.width);


				circle(frame, pupilaEsquerda, 3, Scalar(0, 0, 255));
				circle(frame, pupilaDireita, 3, Scalar(0, 0, 255));

				rectangle(frame, result, Scalar(0, 0, 255), 2, 0);
				rectangle(frame, olhoDireito, Scalar(0, 255, 255), 2, 0);
				rectangle(frame, olhoEsquerdo, Scalar(0, 255, 255), 2, 0);

				if (countTime <= 25) {

					calibrates(frame);

				}

				else {

					calcNewGaze();
					setCursorPosition(deltas.x, deltas.y);
					gazeA = gaze;
				}

			}

			imshow("teste", frame);


			char c = (char)waitKey(1);

			if (c == 27)
				break;
		}

	}

	return 0;
}


Rect detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip)
{


	double fx = 1 / scale;
	Rect test;
	double t = 0;
	vector<Rect> faces, faces2;
	Mat gray, smallImg;

	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};

	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);


	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0, Size(30, 30));

	if (tryflip)
	{
		flip(smallImg, smallImg, 1); 
		cascade.detectMultiScale(smallImg, faces2, 1.1, 2, 0, Size(30, 30));


		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}

	t = (double)getTickCount() - t;



	for (size_t i = 0; i < faces.size(); i++)
	{

		r = faces[i];

		Mat smallImgROI, centerEyesImg;
		vector<Rect> nestedObjects;
		Point center, center2;
		Scalar color = colors[i % 8];
		int radius2 = 0;
		int radius;
		Mat novo;
		vector<Vec3f> circles;
		double aspect_ratio = (double)r.width / r.height;


		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);

		}

		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)), cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)), color, 3, 8, 0);

		if (nestedCascade.empty())
			continue;

		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2, 0, Size(30, 30));


		for (size_t j = 0; j < nestedObjects.size(); j++)
		{

			nr = nestedObjects[j];

			center2.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center2.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius2 = cvRound((nr.width + nr.height)*0.25*scale);
			test = Rect(r.x, center2.y - radius2, r.width, radius2 * 2);


		}


	}

	return test;


}


void Frames_seg() {

	time_t now = time(0);
	novo_seg = now % 100;
	frames += 1;

	if (novo_seg != seg) {
		seg = novo_seg;
		//cout << frames << endl;
		frames = 0;
	}

}


void detectEyesCorners(Mat img) {


	Mat boxFace = img.clone();
	cvtColor(boxFace, boxFace, CV_BGR2GRAY);
	IplImage* src_grayIPL = cvCloneImage(&(IplImage)boxFace);
	FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");
	int bbox[] = { r.x, r.y, r.width + r.x, r.height + r.y };
	double * landmarks = (double*)malloc(2 * model->data.options.M * sizeof(double));
	flandmark_detect(src_grayIPL, bbox, model, landmarks);

	for (int i = 2; i < 2 * model->data.options.M; i += 2)
	{

		if (int(landmarks[i + 1]) < (r.y + nr.y + nr.height)) {
			eyesCorners.push_back(Point2f(int(landmarks[i]), int(landmarks[i + 1])));
		}

	}


}


void setCursorPosition(int deltaXcursor, int deltaYcursor) {

	HWND hwndFound;

	//LPWSTR a = L"pszNewWindowTitle";
	//LPWSTR b = L"pszOldWindowTitle";
 
	char pszNewWindowTitle[MY_BUFSIZE]; // Contains fabricated WindowTitle.
	char pszOldWindowTitle[MY_BUFSIZE]; // Contains original WindowTitle.
	
	int x = GetSystemMetrics(SM_CXFULLSCREEN);
	int y = GetSystemMetrics(SM_CYFULLSCREEN);
	float scalarX = x / frame.cols;
	float scalarY = y / frame.rows;

	//cout << scalarX << endl;
	//cout << scalarY << endl;

	GetConsoleTitle(pszOldWindowTitle, MY_BUFSIZE);
	SetConsoleTitle(pszNewWindowTitle);
	hwndFound = FindWindow(NULL, pszNewWindowTitle);

	if (hwndFound)
	{
		if (flag2 == 0) {
			cursor.right = x / 2;
			cursor.bottom = y / 2;
			SetCursorPos(cursor.right, cursor.bottom);
			flag2 = 1;
		}


		else {
			if (gaze != gazeA) {

				cursor.right = cursor.right + (((gazeA.x - gaze.x)*(x / deltaXcursor))*scalarX);
				//cursor.right = cursor.right + (((gazeA.x - gaze.x)*deltaXcursor)*scalarX);
				//cursor.bottom = cursor.bottom - (((gazeA.y - gaze.y)*deltaYcursor)*scalarY);
				cursor.bottom = cursor.bottom - (((gazeA.y - gaze.y)*(y / deltaYcursor))*scalarY);
				SetCursorPos(cursor.right, cursor.bottom);

			}
		}
	}

}


void calibrates(Mat frame) {
	

	int x = GetSystemMetrics(SM_CXFULLSCREEN);
	int y = GetSystemMetrics(SM_CYFULLSCREEN);
	imageCab = Mat(y, x, CV_8UC3, Scalar(0, 0, 0));
	
	if (frames == 0) {
		countTime++;
	}

	//circle(imageCab, Point(8, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
	//circle(imageCab, Point(x - 11, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
	//circle(imageCab, Point(x - 11, y - 11), 3, Scalar(0, 0, 255), 4, 8, 0);
	//circle(imageCab, Point(8, y - 11), 3, Scalar(0, 0, 255), 4, 8, 0);
	//circle(imageCab, Point(x / 2, y / 2), 3, Scalar(0, 0, 255), 4, 8, 0);

	//imshow("EEny", imageCab);
	
	switch (countTime) {
	
		//ponto superior esquerdo
		case 1:
		case 2:
		case 3:
		case 4:
			circle(imageCab, Point(8, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
			break;
	
		case 5:
			circle(imageCab, Point(8, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
	
			if (initialized) {
				calibrateEL.push_back(Point(pupilaEsquerda.x, pupilaEsquerda.y));
				calibrateER.push_back(Point(pupilaDireita.x, pupilaDireita.y));
				cout << "CANTO SUPERIOR ESQUERDO: Posicao pupila olho esquerdo:" << endl;
				cout << Point(pupilaEsquerda.x, pupilaEsquerda.y) << endl;
				cout << "CANTO SUPERIOR ESQUERDO: Posicao pupila olho direito:" << endl;
				cout << Point(pupilaDireita.x, pupilaDireita.y) << endl;
				initialized = false;
			}
	
			break;
	
		//ponto superior direito
		case 6:
		case 7:
		case 8:
		case 9:
			initialized = true;
			circle(imageCab, Point(x-8, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
			break;
	
		case 10:
			circle(imageCab, Point(x-8, 8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
	
			if (initialized) {
				calibrateEL.push_back(Point(pupilaEsquerda.x, pupilaEsquerda.y));
				calibrateER.push_back(Point(pupilaDireita.x, pupilaDireita.y));
				cout << "CANTO SUPERIOR DIREITO: Posicao pupila olho esquerdo:" << endl;
				cout << Point(pupilaEsquerda.x, pupilaEsquerda.y) << endl;
				cout << "CANTO SUPERIOR DIREITO: Posicao pupila olho direito:" << endl;
				cout << Point(pupilaDireita.x, pupilaDireita.y) << endl;
				initialized = false;
			}
			break;
	
	
		//ponto inferior direito
		case 11:
		case 12:
		case 13:
		case 14:
			initialized = true;
			circle(imageCab, Point(x-8, y-8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
			break;
	
		case 15:
			circle(imageCab, Point(x-8, y-8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
	
			if (initialized) {
				calibrateEL.push_back(Point(pupilaEsquerda.x, pupilaEsquerda.y));
				calibrateER.push_back(Point(pupilaDireita.x, pupilaDireita.y));
				cout << "CANTO INFERIOR DIREITO: Posicao pupila olho esquerdo:" << endl;
				cout << Point(pupilaEsquerda.x, pupilaEsquerda.y) << endl;
				cout << "CANTO INFERIOR DIREITO: Posicao pupila olho direito:" << endl;
				cout << Point(pupilaDireita.x, pupilaDireita.y) << endl;
				initialized = false;
			}
			break;
	
	
		//ponto inferior esquerdo
		case 16:
		case 17:
		case 18:
		case 19:
			initialized = true;
			circle(imageCab, Point(8, y-8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
			break;
	
		case 20:
			circle(imageCab, Point(8, y-8), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
	
			if (initialized) {
				calibrateEL.push_back(Point(pupilaEsquerda.x, pupilaEsquerda.y));
				calibrateER.push_back(Point(pupilaDireita.x, pupilaDireita.y));
				cout << "CANTO INFERIOR ESQUERDO: Posicao pupila olho esquerdo:" << endl;
				cout << Point(pupilaEsquerda.x, pupilaEsquerda.y) << endl;
				cout << "CANTO INFERIOR ESQUERDO: Posicao pupila olho direito:" << endl;
				cout << Point(pupilaDireita.x, pupilaDireita.y) << endl;
				initialized = false;
			}
			break;
	
		//ponto central
		case 21:
		case 22:
		case 23:
		case 24:
			initialized = true;
			circle(imageCab, Point(x/2, y/2), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
			break;
	
		case 25:
			circle(imageCab, Point(x/2, y/2), 3, Scalar(0, 0, 255), 4, 8, 0);
			imshow("EEny", imageCab);
	
			if (initialized) {
				calibrateEL.push_back(Point(pupilaEsquerda.x, pupilaEsquerda.y));
				calibrateER.push_back(Point(pupilaDireita.x, pupilaDireita.y));
				cout << "CENTRO: Posicao pupila olho esquerdo:" << endl;
				cout << Point(pupilaEsquerda.x, pupilaEsquerda.y) << endl;
				cout << "CENTRO: Posicao pupila olho direito:" << endl;
				cout << Point(pupilaDireita.x, pupilaDireita.y) << endl;
				initialized = false;
	
				calibrateEyes.push_back(calibrateEL);
				calibrateEyes.push_back(calibrateER);
				calcDeltaX_DeltaY();
			}
			break;
	
		default:
			destroyWindow("EEny");
			break;
	}

}


void calcDeltaX_DeltaY() {
	float deltaX = 0, deltaY = 0;

	vector<Point> calcEL;
	calcEL = calibrateEyes[0];
	vector<Point> calcER;
	calcER = calibrateEyes[1];

	Point upLEL = calcEL[0];
	Point upLER = calcER[0];
	Point upREL = calcEL[1];
	Point upRER = calcER[1];
	Point bottomLEL = calcEL[3];
	Point bottomLER = calcER[3];
	Point bottomREL = calcEL[2];
	Point bottomRER = calcER[2];
	
	//deltaX = (((upLEL.x + upLER.x) / 2) - ((upREL.x + upRER.x)/2));
	deltaX = (abs((((upLEL.x + upLER.x) / 2) - ((upREL.x + upRER.x) / 2))) + abs((((bottomLEL.x + bottomLER.x) / 2) - ((bottomREL.x + bottomRER.x) / 2)))) / 2;
	//deltaX = (abs((((upLEL.x + upLER.x) / 2) - ((upREL.x + upRER.x) / 2))) + abs((((bottomLEL.x + bottomLER.x) / 2) - ((bottomREL.x + bottomRER.x) / 2))));
	

	//deltaY = abs(((((upLEL.y + upLER.y) / 2) - ((bottomLEL.y + bottomLER.y) / 2)) + (((upREL.y + upRER.y) / 2) - ((bottomREL.y + bottomRER.y) / 2))));
	deltaY = (abs((((upLEL.y + upLER.y) / 2) - ((bottomLEL.y + bottomLER.y) / 2))) + abs((((upREL.y + upRER.y) / 2) - ((bottomREL.y + bottomRER.y) / 2)))) / 2;
	//deltaY = (abs((((upLEL.y + upLER.y) / 2) - ((bottomLEL.y + bottomLER.y) / 2))) + abs((((upREL.y + upRER.y) / 2) - ((bottomREL.y + bottomRER.y) / 2))));
	
	wcout <<"Delta x: " << deltaX << endl;
	wcout <<"Delta y: " << deltaY << endl;
	

	deltas = Point(deltaX, deltaY);

}


void calcNewGaze() {

	gaze = Point((pupilaEsquerda.x + pupilaDireita.x) / 2, (pupilaEsquerda.y + pupilaDireita.y) / 2);
	circle(frame, gaze, 4, Scalar(255, 0, 0),4, 8, 0);

}