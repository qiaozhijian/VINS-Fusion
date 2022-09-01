#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

namespace camodocal
{

#if (CV_VERSION_MAJOR >= 4)
#define CV_BGR2RGB                          cv::COLOR_BGR2RGB
#define CV_GRAY2RGB                         cv::COLOR_GRAY2RGB
#define CV_GRAY2BGR                         cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY                         cv::COLOR_BGR2GRAY
#define CV_AA                               cv::LINE_AA
#define CV_CALIB_CB_ADAPTIVE_THRESH         cv::CALIB_CB_ADAPTIVE_THRESH
#define CV_CALIB_CB_NORMALIZE_IMAGE         cv::CALIB_CB_NORMALIZE_IMAGE
#define CV_CALIB_CB_FILTER_QUADS            cv::CALIB_CB_FILTER_QUADS
#define CV_CALIB_CB_FAST_CHECK              cv::CALIB_CB_FAST_CHECK
#define CV_ADAPTIVE_THRESH_MEAN_C           cv::ADAPTIVE_THRESH_MEAN_C
#define CV_THRESH_BINARY                    cv::THRESH_BINARY
#define CV_SHAPE_CROSS                      cv::MORPH_CROSS
#define CV_SHAPE_RECT                       cv::MORPH_RECT
#define CV_TERMCRIT_EPS                     cv::TermCriteria::EPS
#define CV_RETR_CCOMP                       cv::RETR_CCOMP
#define CV_CHAIN_APPROX_SIMPLE              cv::CHAIN_APPROX_SIMPLE
#define CV_THRESH_BINARY_INV                cv::CHAIN_APPROX_SIMPLE
#define CV_TERMCRIT_ITER                    cv::TermCriteria::MAX_ITER
#endif

// forward declarations
class ChessboardCorner;
typedef boost::shared_ptr<ChessboardCorner> ChessboardCornerPtr;
class ChessboardQuad;
typedef boost::shared_ptr<ChessboardQuad> ChessboardQuadPtr;

class Chessboard
{
public:
    Chessboard(cv::Size boardSize, cv::Mat& image);

    void findCorners(bool useOpenCV = false);
    const std::vector<cv::Point2f>& getCorners(void) const;
    bool cornersFound(void) const;

    const cv::Mat& getImage(void) const;
    const cv::Mat& getSketch(void) const;

private:
    bool findChessboardCorners(const cv::Mat& image,
                               const cv::Size& patternSize,
                               std::vector<cv::Point2f>& corners,
                               int flags, bool useOpenCV);

    bool findChessboardCornersImproved(const cv::Mat& image,
                                       const cv::Size& patternSize,
                                       std::vector<cv::Point2f>& corners,
                                       int flags);

    void cleanFoundConnectedQuads(std::vector<ChessboardQuadPtr>& quadGroup, cv::Size patternSize);

    void findConnectedQuads(std::vector<ChessboardQuadPtr>& quads,
                            std::vector<ChessboardQuadPtr>& group,
                            int group_idx, int dilation);

//    int checkQuadGroup(std::vector<ChessboardQuadPtr>& quadGroup,
//                       std::vector<ChessboardCornerPtr>& outCorners,
//                       cv::Size patternSize);

    void labelQuadGroup(std::vector<ChessboardQuadPtr>& quad_group,
                        cv::Size patternSize, bool firstRun);

    void findQuadNeighbors(std::vector<ChessboardQuadPtr>& quads, int dilation);

    int augmentBestRun(std::vector<ChessboardQuadPtr>& candidateQuads, int candidateDilation,
                       std::vector<ChessboardQuadPtr>& existingQuads, int existingDilation);

    void generateQuads(std::vector<ChessboardQuadPtr>& quads,
                       cv::Mat& image, int flags,
                       int dilation, bool firstRun);

    bool checkQuadGroup(std::vector<ChessboardQuadPtr>& quads,
                        std::vector<ChessboardCornerPtr>& corners,
                        cv::Size patternSize);

    void getQuadrangleHypotheses(const std::vector< std::vector<cv::Point> >& contours,
                                 std::vector< std::pair<float, int> >& quads,
                                 int classId) const;

    bool checkChessboard(const cv::Mat& image, cv::Size patternSize) const;

    bool checkBoardMonotony(std::vector<ChessboardCornerPtr>& corners,
                            cv::Size patternSize);

    bool matchCorners(ChessboardQuadPtr& quad1, int corner1,
                      ChessboardQuadPtr& quad2, int corner2) const;

    cv::Mat mImage;
    cv::Mat mSketch;
    std::vector<cv::Point2f> mCorners;
    cv::Size mBoardSize;
    bool mCornersFound;
};

}

#endif
