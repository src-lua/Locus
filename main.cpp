#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

cv::Mat normalizePoints(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& normalizedPoints) {
    float meanX = 0, meanY = 0; for (const auto& p : points) { meanX += p.x; meanY += p.y; }
    meanX /= points.size(); meanY /= points.size(); float meanDist = 0;
    normalizedPoints.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        normalizedPoints[i].x = points[i].x - meanX; normalizedPoints[i].y = points[i].y - meanY;
        meanDist += std::sqrt(normalizedPoints[i].x * normalizedPoints[i].x + normalizedPoints[i].y * normalizedPoints[i].y);
    }
    meanDist /= points.size(); float scale = std::sqrt(2.0f) / meanDist;
    for (auto& p : normalizedPoints) { p.x *= scale; p.y *= scale; }
    cv::Mat T = (cv::Mat_<float>(3, 3) << scale, 0, -scale * meanX, 0, scale, -scale * meanY, 0, 0, 1);
    return T;
}

cv::Mat computeHomographyDLT(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    if (points1.size() < 4) { return cv::Mat(); }
    std::vector<cv::Point2f> normPoints1, normPoints2;
    cv::Mat T1 = normalizePoints(points1, normPoints1); cv::Mat T2 = normalizePoints(points2, normPoints2);
    cv::Mat A(2 * points1.size(), 9, CV_32F);
    for (size_t i = 0; i < points1.size(); ++i) {
        float u1 = normPoints1[i].x; float v1 = normPoints1[i].y; float u2 = normPoints2[i].x; float v2 = normPoints2[i].y;
        A.at<float>(2*i,0)=-u1;A.at<float>(2*i,1)=-v1;A.at<float>(2*i,2)=-1;A.at<float>(2*i,3)=0; A.at<float>(2*i,4)=0;  A.at<float>(2*i,5)=0;
        A.at<float>(2*i,6)=u1*u2; A.at<float>(2*i,7)=v1*u2; A.at<float>(2*i,8)=u2;
        A.at<float>(2*i+1,0)=0;  A.at<float>(2*i+1,1)=0;  A.at<float>(2*i+1,2)=0; A.at<float>(2*i+1,3)=-u1;A.at<float>(2*i+1,4)=-v1;A.at<float>(2*i+1,5)=-1;
        A.at<float>(2*i+1,6)=u1*v2;A.at<float>(2*i+1,7)=v1*v2;A.at<float>(2*i+1,8)=v2;
    }
    cv::SVD svd(A, cv::SVD::FULL_UV); cv::Mat h = svd.vt.row(8).t();
    cv::Mat H_norm = h.reshape(0, 3); cv::Mat H = T2.inv() * H_norm * T1;
    H /= H.at<float>(2, 2); return H;
}

class HomographyLMSolverCallback : public cv::LMSolver::Callback {
public:
    HomographyLMSolverCallback(const std::vector<cv::Point2f>& p1, const std::vector<cv::Point2f>& p2) : points1_h(p1), points2_h(p2) {}
    bool compute(cv::InputArray param, cv::OutputArray err, cv::OutputArray J) const override {
        cv::Mat h = param.getMat();
        cv::Mat H = (cv::Mat_<double>(3,3) << h.at<double>(0),h.at<double>(1),h.at<double>(2),h.at<double>(3),h.at<double>(4),h.at<double>(5),h.at<double>(6),h.at<double>(7),1.0);
        int n_points = points1_h.size(); err.create(2 * n_points, 1, CV_64F); cv::Mat errMat = err.getMat();
        if(J.needed()){ J.create(2*n_points,8,CV_64F); }
        for (int i = 0; i < n_points; ++i) {
            double u1=points1_h[i].x; double v1=points1_h[i].y; double u2_obs=points2_h[i].x; double v2_obs=points2_h[i].y;
            double w_proj = H.at<double>(2,0)*u1+H.at<double>(2,1)*v1+1.0;
            double u2_proj = (H.at<double>(0,0)*u1+H.at<double>(0,1)*v1+H.at<double>(0,2))/w_proj;
            double v2_proj = (H.at<double>(1,0)*u1+H.at<double>(1,1)*v1+H.at<double>(1,2))/w_proj;
            errMat.at<double>(2*i,0)=u2_proj-u2_obs; errMat.at<double>(2*i+1,0)=v2_proj-v2_obs;
            if(J.needed()){
                cv::Mat J_i=J.getMat().rowRange(2*i,2*i+2); double inv_w_proj=1.0/w_proj;
                J_i.at<double>(0,0)=u1*inv_w_proj;J_i.at<double>(0,1)=v1*inv_w_proj;J_i.at<double>(0,2)=inv_w_proj;
                J_i.at<double>(0,3)=0;J_i.at<double>(0,4)=0;J_i.at<double>(0,5)=0;
                J_i.at<double>(0,6)=-u1*u2_proj*inv_w_proj;J_i.at<double>(0,7)=-v1*u2_proj*inv_w_proj;
                J_i.at<double>(1,0)=0;J_i.at<double>(1,1)=0;J_i.at<double>(1,2)=0;
                J_i.at<double>(1,3)=u1*inv_w_proj;J_i.at<double>(1,4)=v1*inv_w_proj;J_i.at<double>(1,5)=inv_w_proj;
                J_i.at<double>(1,6)=-u1*v2_proj*inv_w_proj;J_i.at<double>(1,7)=-v1*v2_proj*inv_w_proj;
            }
        } return true;
    }
private: std::vector<cv::Point2f> points1_h, points2_h;
};


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // --- Etapa 1: Pré-Processamento ---
    cv::Mat img1_color = cv::imread("img1.jpg"), img2_color = cv::imread("img2.jpg");
    if (img1_color.empty() || img2_color.empty()) { return -1; }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1_color, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2_color, cv::noArray(), keypoints2, descriptors2);

    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // --- Etapa 2: Filtragem ---
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> all_matches;
    const float ratio_thresh = 0.75f;
    for (const auto& match_pair : knn_matches) {
        if (!match_pair.empty()) {
            all_matches.push_back(match_pair[0]);
            if (match_pair.size() > 1 && match_pair[0].distance < ratio_thresh * match_pair[1].distance) {
                good_matches.push_back(match_pair[0]);
            }
        }
    }
    
    // --- Etapa 3: Saída no Terminal (Limpa) ---
    std::cout << "--- Analise das Correspondencias ---" << std::endl;
    std::cout << "Correspondencias Boas (filtradas): " << good_matches.size() << std::endl;
    std::cout << "Correspondencias Ruins (descartadas): " << all_matches.size() - good_matches.size() << std::endl;

    // --- Etapa 4: Visualizações das Correspondências ---
    cv::Mat img_good_matches, img_all_matches;
    cv::drawMatches(img1_color, keypoints1, img2_color, keypoints2, good_matches, img_good_matches);
    cv::drawMatches(img1_color, keypoints1, img2_color, keypoints2, all_matches, img_all_matches);
    cv::namedWindow("Boas Correspondencias", cv::WINDOW_NORMAL);
    cv::imshow("Boas Correspondencias", img_good_matches);
    cv::namedWindow("Todas as Correspondencias (Sem Filtro)", cv::WINDOW_NORMAL);
    cv::imshow("Todas as Correspondencias (Sem Filtro)", img_all_matches);

    // --- Etapa 5: Cálculo da Homografia ---
    if (good_matches.size() < 4) {
        std::cout << "\nNao ha correspondencias boas suficientes para calcular a homografia." << std::endl;
        cv::waitKey(0);
        return -1;
    }
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    cv::Mat H_dlt = computeHomographyDLT(points1, points2);
    
    H_dlt.convertTo(H_dlt, CV_64F);
    cv::Mat params(8, 1, CV_64F);
    for (int i=0; i<8; ++i) params.at<double>(i) = H_dlt.at<double>(i/3, i%3);
    auto callback = cv::makePtr<HomographyLMSolverCallback>(points1, points2);
    auto solver = cv::LMSolver::create(callback, 100);
    solver->run(params);
    cv::Mat H_refined = (cv::Mat_<double>(3, 3) << params.at<double>(0),params.at<double>(1),params.at<double>(2),params.at<double>(3),params.at<double>(4),params.at<double>(5),params.at<double>(6),params.at<double>(7),1.0);

    std::cout << "\n--- Matriz de Homografia (H) Inicial (DLT) ---" << std::endl;
    std::cout << H_dlt << std::endl;
    std::cout << "\n--- Matriz de Homografia (H) Refinada (LM) ---" << std::endl;
    std::cout << H_refined << std::endl;

    // --- Etapa 6: Visualização Final do Alinhamento (Completa) ---
    cv::Mat warped_dlt, warped_refined;
    cv::warpPerspective(img1_color, warped_dlt, H_dlt, img2_color.size());
    cv::warpPerspective(img1_color, warped_refined, H_refined, img2_color.size());

    cv::namedWindow("Imagem 2 Original", cv::WINDOW_NORMAL);
    cv::imshow("Imagem 2 Original", img2_color);
    cv::namedWindow("Alinhada com DLT", cv::WINDOW_NORMAL);
    cv::imshow("Alinhada com DLT", warped_dlt);
    cv::namedWindow("Alinhada e Refinada com LM", cv::WINDOW_NORMAL);
    cv::imshow("Alinhada e Refinada com LM", warped_refined);
    
    cv::waitKey(0);
    return 0;
}