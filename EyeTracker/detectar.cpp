#include "Header Files/detectar.h"

Rect detectar::Face(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip)
{
	Rect r;
	int indice = 0, maior = NULL;
	double fx = 1 / scale;
	Rect test;
	vector<Rect> faces;
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{

		r = faces[i];

		rectangle(img, r, Scalar(1), 3, 8, 0);

		if (nestedCascade.empty())
			continue;


	}

	return r;

}

vector<Rect> detectar::Olho(Mat img, CascadeClassifier& nestedCascade) {

	vector<Rect> nestedObjects, vetor;

	Rect total, olhoEsquerdo, olhoDireito, nr;

	nestedCascade.detectMultiScale(img, nestedObjects, 1.1, 2, 0, Size(30, 30));

	if (nestedObjects.size() == 2) {
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			nr = nestedObjects[j];

			if (nr.x < (img.cols) / 2) {
				olhoEsquerdo = nr;
				rectangle(img, nr, Scalar(255, 0, 0), 3);
			}
			else {
				olhoDireito = nr;
				rectangle(img, nr, Scalar(0, 255, 0), 3);
			}

		}
		total.x = olhoEsquerdo.x - 20;

		if (olhoEsquerdo.y < olhoDireito.y) {
			total.y = olhoEsquerdo.y;
			total.height = olhoEsquerdo.height;
		}
		else {
			total.y = olhoDireito.y;
			total.height = olhoDireito.height;
		}

		total.width = olhoDireito.x + olhoDireito.width - olhoEsquerdo.x + 40;

		cv::imshow("Total", img);
		vetor = { total, olhoEsquerdo, olhoDireito };
	}
	return vetor;


}


vector<Rect> detectar::OlhoProporcao(Rect face) {


	int olho_width = (int)(face.width * (35 / 100.0));
	int olho_height = (int)(face.width * (30 / 100.0));
	int olho_top = (int)(face.height * (25 / 100.0));
	Rect olhoEsquerdo((int)(face.width*(13 / 100.0)), olho_top, olho_width, olho_height);
	Rect olhoDireito((int)(face.width - olho_width - face.width*(13 / 100.0)), olho_top, olho_width, olho_height);

	Rect rectOlhoEsquerdo;
	rectOlhoEsquerdo.x = olhoEsquerdo.x;
	rectOlhoEsquerdo.height = olhoEsquerdo.height / 2;
	rectOlhoEsquerdo.y = olhoEsquerdo.y;
	rectOlhoEsquerdo.y += rectOlhoEsquerdo.height / 2;
	rectOlhoEsquerdo.width = olho_width;

	Rect rectOlhoDireito;
	rectOlhoDireito.x = olhoDireito.x;
	rectOlhoDireito.width = olho_width;
	rectOlhoDireito.height = rectOlhoEsquerdo.height;
	rectOlhoDireito.y = rectOlhoEsquerdo.y;

	Rect total;
	total.x = olhoEsquerdo.x;
	total.width = olhoDireito.x + olhoDireito.width - total.x;
	total.y = rectOlhoEsquerdo.y;
	total.height = rectOlhoEsquerdo.height;

	vector<Rect> vetor = { total, rectOlhoEsquerdo , rectOlhoDireito };

	return vetor;

}