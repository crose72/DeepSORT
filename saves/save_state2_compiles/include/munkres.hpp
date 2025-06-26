#ifndef MUNKRES_H
#define MUNKRES_H
#include <opencv2/opencv.hpp>

class Munkres
{
public:
    Munkres()
    {
    }
    std::vector<std::pair<int, int>> compute(cv::Mat &cost_matrix);

private:
    // variables
    cv::Mat C;
    int n;
    int Z0_r = 0;
    int Z0_c = 0;
    int original_length;
    int original_width;
    std::vector<bool> row_covered;
    std::vector<bool> col_covered;
    cv::Mat path;
    cv::Mat marked;

    //
    int run_step(int step);
    cv::Mat pad_matrix(cv::Mat &matrix, int pad_value = 0);
    cv::Mat make_matrix(int n, int val);
    void find_a_zero(int &row, int &col);
    int find_star_in_row(int row);
    int find_star_in_col(int col);
    int find_prime_in_row(int row);
    int find_smallest();
    void convert_path(cv::Mat &path, int count);
    void clear_covers();
    void erase_primes();
    int step1();
    int step2();
    int step3();
    int step4();
    int step5();
    int step6();
};
#endif // !MUNKRES_H
