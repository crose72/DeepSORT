#include "munkres.hpp"

/*
Compute the indexes for the lowest-cost pairings between rows and
columns in the database. Returns a list of (row, column) tuples
that can be used to traverse the matrix.

:Parameters:
cost_matrix : cv::Mat
The cost matrix. If this cost matrix is not square, it
will be padded with zeros, via a call to ``pad_matrix()``.
(This method does *not* modify the caller's matrix. It
operates on a copy of the matrix.)

**WARNING**: This code handles square and rectangular
matrices. It does *not* handle irregular matrices.

:rtype: list
:return: A list of ``(row, column)`` tuples that describe the lowest
cost path through the matrix
*/
std::vector<std::pair<int, int>> Munkres::compute(cv::Mat &cost_matrix)
{

    C = pad_matrix(cost_matrix);
    n = C.rows;
    original_length = cost_matrix.rows;
    original_width = cost_matrix.cols;
    row_covered = std::vector<bool>(n, false);
    col_covered = std::vector<bool>(n, false);
    Z0_r = 0;
    Z0_c = 0;
    path = make_matrix(n * 2, 0);
    marked = make_matrix(n, 0);

    int step = 1;

    while (true)
    {
        // printf("step %d\n", step);
        step = run_step(step);
        if (!(step >= 0 && step <= 6))
            break;
    }

    std::vector<std::pair<int, int>> results;
    for (int i = 0; i < original_length; i++)
    {
        for (int j = 0; j < original_width; j++)
        {
            if (marked.at<int>(i, j) == 1)
            {
                results.push_back(std::pair<int, int>(i, j));
            }
        }
    }
    return results;
}

/*
Pad a possibly non-square matrix to make it square.

:Parameters:
matrix : cv::Mat
matrix to pad

pad_value : int
value to use to pad the matrix

:rtype: list of lists
:return: a new, possibly padded, matrix
*/
cv::Mat Munkres::pad_matrix(cv::Mat &matrix, int pad_value)
{

    int cols = matrix.cols;
    int rows = matrix.rows;
    if (cols == rows)
        return matrix.clone();
    int max_sz = std::max(cols, rows);
    cv::Mat_<int> new_matrix(max_sz, max_sz);
    new_matrix.setTo(pad_value);
    matrix.copyTo(new_matrix(cv::Range(0, rows), cv::Range(0, cols)));
    return new_matrix;
}

int Munkres::run_step(int step)
{
    switch (step)
    {
    case 1:
        return step1();
    case 2:
        return step2();
    case 3:
        return step3();
    case 4:
        return step4();
    case 5:
        return step5();
    case 6:
        return step6();
    default:
        return -1;
    }
}

cv::Mat Munkres::make_matrix(int n, int val)
{
    cv::Mat_<int> new_mat(n, n);
    new_mat.setTo(val);
    return new_mat;
}

/*
For each row of the matrix, find the smallest element and
subtract it from every element in its row. Go to Step 2.
*/
int Munkres::step1()
{

    for (int i = 0; i < C.rows; i++)
    {
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::minMaxLoc(C.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        C.row(i) -= static_cast<int>(minVal);
    }

    return 2;
}

/*
Find a zero (Z) in the resulting matrix. If there is no starred
zero in its row or column, star Z. Repeat for each element in the
matrix. Go to Step 3.
*/
int Munkres::step2()
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if ((C.at<int>(i, j) == 0) && (!col_covered[j]) && (!row_covered[i]))
            {
                marked.at<int>(i, j) = 1;
                col_covered[j] = true;
                row_covered[i] = true;
            }
        }
    }
    clear_covers();
    return 3;
}

/*
Cover each column containing a starred zero. If K columns are
covered, the starred zeros describe a complete set of unique
assignments. In this case, Go to DONE, otherwise, Go to Step 4.
*/
int Munkres::step3()
{
    int count = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (marked.at<int>(i, j) == 1)
            {
                col_covered[j] = true;
                count++;
            }
        }
    }
    int step = -1;
    if (count >= n)
        step = 7; // done
    else
        step = 4;
    return step;
}

/*
Find a noncovered zero and prime it. If there is no starred zero
in the row containing this primed zero, Go to Step 5. Otherwise,
cover this row and uncover the column containing the starred
zero. Continue in this manner until there are no uncovered zeros
left. Save the smallest uncovered value and Go to Step 6.
*/
int Munkres::step4()
{

    int step = 0;
    bool done = false;
    int row = -1;
    int col = -1;
    int star_col = -1;
    while (!done)
    {
        find_a_zero(row, col);
        if (row < 0)
        {
            done = true;
            step = 6;
        }
        else
        {
            marked.at<int>(row, col) = 2;
            int star_col = find_star_in_row(row);
            if (star_col >= 0)
            {
                col = star_col;
                row_covered[row] = true;
                col_covered[col] = false;
            }
            else
            {
                done = true;
                Z0_r = row;
                Z0_c = col;
                step = 5;
            }
        }
    }

    return step;
}

/*
Construct a series of alternating primed and starred zeros as
follows. Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0 (if any).
Let Z2 denote the primed zero in the row of Z1 (there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred zero
of the series, star each primed zero of the series, erase all
primes and uncover every line in the matrix. Return to Step 3
*/
int Munkres::step5()
{
    int count = 0;
    path.at<int>(count, 0) = Z0_r;
    path.at<int>(count, 1) = Z0_c;
    bool done = false;
    while (!done)
    {
        int row = find_star_in_col(path.at<int>(count, 1));
        if (row >= 0)
        {
            count++;
            path.at<int>(count, 0) = row;
            path.at<int>(count, 1) = path.at<int>(count - 1, 1);
        }
        else
        {
            done = true;
        }
        if (!done)
        {
            int col = find_prime_in_row(path.at<int>(count, 0));
            count++;
            path.at<int>(count, 0) = path.at<int>(count - 1, 0);
            path.at<int>(count, 1) = col;
        }
    }
    convert_path(path, count);
    clear_covers();
    erase_primes();
    return 3;
}

/*
Add the value found in Step 4 to every element of each covered
row, and subtract it from every element of each uncovered column.
Return to Step 4 without altering any stars, primes, or covered
lines.
*/
int Munkres::step6()
{
    int minval = find_smallest();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (row_covered[i])
                C.at<int>(i, j) += minval;
            if (!col_covered[j])
                C.at<int>(i, j) -= minval;
        }
    }
    return 4;
}

void Munkres::convert_path(cv::Mat &path, int count)
{
    for (int i = 0; i < count + 1; i++)
    {
        if (marked.at<int>(path.at<int>(i, 0), path.at<int>(i, 1)) == 1)
            marked.at<int>(path.at<int>(i, 0), path.at<int>(i, 1)) = 0;
        else
            marked.at<int>(path.at<int>(i, 0), path.at<int>(i, 1)) = 1;
    }
}

/*
Clear all covered matrix cells
*/
void Munkres::clear_covers()
{
    for (int i = 0; i < n; i++)
    {
        row_covered[i] = false;
        col_covered[i] = false;
    }
}

/*
Erase all prime markings
*/
void Munkres::erase_primes()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (marked.at<int>(i, j) == 2)
            {
                marked.at<int>(i, j) = 0;
            }
        }
    }
}

/*
Find the first uncovered element with value 0
*/
void Munkres::find_a_zero(int &row, int &col)
{

    row = -1;
    col = -1;
    int i = 0;
    bool done = false;
    while (!done)
    {
        int j = 0;
        while (true)
        {
            if ((C.at<int>(i, j) == 0) && (!row_covered[i]) && (!col_covered[j]))
            {
                row = i;
                col = j;
                done = true;
            }
            j++;
            if (j >= n)
                break;
        }
        i++;
        if (i >= n)
            done = true;
    }
}

/*
Find the first starred element in the specified row. Returns
the column index, or -1 if no starred element was found.
*/
int Munkres::find_star_in_row(int row)
{

    int col = -1;
    for (int j = 0; j < n; j++)
    {
        if (marked.at<int>(row, j) == 1)
        {
            col = j;
            break;
        }
    }
    return col;
}

/*
Find the first starred element in the specified row. Returns
the row index, or -1 if no starred element was found.
*/
int Munkres::find_star_in_col(int col)
{

    int row = -1;
    for (int i = 0; i < n; i++)
    {
        if (marked.at<int>(i, col) == 1)
        {
            row = i;
            break;
        }
    }
    return row;
}

/*
Find the first prime element in the specified row. Returns
the column index, or -1 if no starred element was found.
*/
int Munkres::find_prime_in_row(int row)
{

    int col = -1;
    for (int j = 0; j < n; j++)
    {
        if (marked.at<int>(row, j) == 2)
        {
            col = j;
            break;
        }
    }
    return col;
}

int Munkres::find_smallest()
{
    int minval = 1e9;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if ((!row_covered[i]) && (!col_covered[j]))
                if (minval > C.at<int>(i, j))
                    minval = C.at<int>(i, j);
        }
    }
    return minval;
}