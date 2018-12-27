#include <iostream>

#include <array>
#include <stack>
#include <vector>

#include <limits>
#include <random>
#include <algorithm>
#include <thread>

#include <iomanip>
#include <chrono>

/* Written by ganler. Use this code to feel the pipelines of diverse sorts. */

// 分别测试：插入排序√，折半插入排序√，希尔排序√，冒泡排序√，快速排序√，选择排序√，归并排序√，堆排序√
// 以及最强排序 STL 排序
// 随机数范围[0, 1000 000]
// 随机数个数10 000个
// 测试次数50次（减少常数的影响）

constexpr uint32_t TEST_SIZE = 10000;
constexpr uint32_t LOOP_SIZE = 10;
constexpr uint32_t PRECISION = 6;
constexpr uint32_t SHELL_FACTOR = 3; // For guys who want use a gap function with $n/SHELL_FACTOR^i$

using namespace std;

void time_test()
{
    auto start = chrono::steady_clock::now();
    this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto end = chrono::steady_clock::now();

    auto diff = end-start;

    cout << "  ==>>  <时间校准>" << endl;
    cout << "程序设计了[1000.00] ms等待，实际等待" << endl << "  ==>>  [" << chrono::duration<double, milli>(diff).count() << "] ms" << endl;
}

void print(const array<int, TEST_SIZE>& arr)
{
    for(const auto& x : arr)
        cout << x << ' ';
    cout << endl;
}

void insert_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();
    auto data = data_;

    for (int i = 0; i < LOOP_SIZE; ++i)
    {
        data = data_;
        for (int j = 1; j < TEST_SIZE; ++j)
        {
            auto tmp = data[j];
            int k = j;
            for (; k > 0 && data[k - 1] > tmp; --k) // Must be "data[k-1] > tmp" not "data[k-1] > data[k]"
                data[k] = data[k - 1];              // "data[k-1] > data[k]" is used in constant swap operation.
            data[k] = tmp;
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[INSERT SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}


void bi_insert_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int i = 0; i < LOOP_SIZE; ++i)
    {// 用swap会慢一点（std::swap有2次操作（手写swap有3次）），直接往后拉就只有一次。
        data = data_;
        for (int j = 1; j < TEST_SIZE; ++j)
        {
            auto tmp = data[j];
            auto beg = 0;
            auto end = j-1;

            while(beg <= end)
            {
                auto mid = (beg+end)/2;
                if(data[mid] == tmp)
                {
                    beg = mid;
                    break;
                }
                else if(data[mid] < tmp)
                    beg = mid+1;
                else
                    end = mid-1;
            }

            for (int k = j; k > beg; --k)
                data[k] = data[k-1];

            data[beg] = tmp;
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[BI_INSERT SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

void shell_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int k = 0; k < LOOP_SIZE; ++k)
    {
        data = data_;
        /* This part is related to the parameter SHELL_FACTOR */
//        for (int gap = TEST_SIZE / SHELL_FACTOR; gap > 0; gap /= SHELL_FACTOR)
//            for (int i = gap; i < TEST_SIZE; ++i)
//                for (int j = i; j - gap >= 0 && data[j - gap] > data[j]; j -= gap)// Must be j>=gap. Especially "=".
//                    swap(data[j], data[j - gap]);
        int step[] = { 1, 5, 19, 41, 109, 209, 505, 929, 2161, 3905 };
        for (int k = 9; k >= 0; k--)
        {
            auto gap = step[k];
            for (int i = gap; i < TEST_SIZE; ++i)
                for (int j = i; j - gap >= 0 && data[j - gap] > data[j]; j -= gap)// Must be j>=gap. Especially "=".
                    swap(data[j], data[j - gap]);
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[SHELL SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

void bubble_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int k = 0; k < LOOP_SIZE; ++k)
    {
        data = data_;
        for (int i = TEST_SIZE-1; i > 0; --i)
            for (int j = 0; j < i; ++j)
                if (data[j] > data[j + 1])
                    swap(data[j], data[j + 1]);
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[BUBBLE SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}


void quick_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;
    random_device rd_;

    for (int i = 0; i < LOOP_SIZE; ++i)
    {
        data = data_;
        stack<pair<uint32_t, uint32_t>> range_stack;
        range_stack.push(pair<uint32_t, uint32_t>(0, TEST_SIZE));

        while (!range_stack.empty())
        {
            auto range = range_stack.top();
            range_stack.pop();

            uint32_t partition = data[range.first];

            uint32_t i = range.first, j;
            for (j = range.first+1; j < range.second; j++)
                if(data[j] <= partition)
                    swap(data[++i], data[j]);

            swap(data[i], data[range.first]);
            if(i > range.first)
                range_stack.push(pair<uint32_t, uint32_t>(range.first, i));
            if(i+1 < range.second)
                range_stack.push(pair<uint32_t, uint32_t>(i+1, range.second));
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[QUICK SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}


void select_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int i = 0; i < LOOP_SIZE; ++i)
    {
        data = data_;
        for (int j = 0; j < TEST_SIZE - 1; ++j)
        {
            auto min_ind = j;
            for (int k = j; k < TEST_SIZE; ++k) {
                if(data[k] < data[min_ind])
                    min_ind = k;
            }
            swap(data[min_ind], data[j]);
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[SELECT SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

void merge(array<int, TEST_SIZE>& arr, uint32_t beg, uint32_t mid, uint32_t end)
{
    auto left_vec = vector<int>(arr.begin()+beg, arr.begin()+mid);
    auto right_vec = vector<int>(arr.begin()+mid, arr.begin()+end);

    uint32_t left_ind = 0;
    uint32_t right_ind = 0;

    left_vec.insert(left_vec.end(), numeric_limits<int>::max());
    right_vec.insert(right_vec.end(), numeric_limits<int>::max());

    for(; beg<end; beg++)
        if(left_vec[left_ind] > right_vec[right_ind])
            arr[beg] = right_vec[right_ind++];
        else
            arr[beg] = left_vec[left_ind++];
}

void merge_sort_(array<int, TEST_SIZE>& arr, int beg=0, int end=TEST_SIZE)
{
    if(beg >= end-1)
        return;
    int mid = beg+(end-beg)/2;
    merge_sort_(arr, beg, mid);
    merge_sort_(arr, mid, end);
    merge(arr, beg, mid, end);
}


void merge_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int i = 0; i < LOOP_SIZE; ++i) {
        data = data_;
        merge_sort_(data);
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[MERGE SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

inline void max_heapify(array<int, TEST_SIZE>& arr, uint32_t begin, uint32_t end)
{
    uint32_t father = begin;
    uint32_t left_son = 2*father+1;

    while(left_son <= end)
    {
        if(left_son+1 <= end && arr[left_son] < arr[left_son+1])
            left_son++;

        if(arr[father] > arr[left_son])
            return;
        else
        {
            swap(arr[father], arr[left_son]);
            father = left_son;
            left_son = 2*father+1;
        }
    }
}

void heap_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    if(TEST_SIZE <= 1){ return; }

    for (int k = 0; k < LOOP_SIZE; ++k) {
        data = data_;
        for (int i = TEST_SIZE / 2 - 1; i >= 0; i--)
            max_heapify(data, i, TEST_SIZE - 1);
        for (int i = TEST_SIZE - 1; i > 0; i--)
        {
            swap(data[0], data[i]);
            max_heapify(data, 0, i - 1);
        }
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[HEAP SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

void std_sort(const array<int, TEST_SIZE>& data_)
{
    auto start = chrono::steady_clock::now();

    auto data = data_;

    for (int i = 0; i < LOOP_SIZE; ++i) {
        data = data_;
        sort(data.begin(), data.end());
    }

    auto end = chrono::steady_clock::now();
    auto diff = (end-start);

    cout << "[STD SORT]: " << endl << "for " << TEST_SIZE << " elemets:" << endl;
    cout << ">>>\t";
    print(data);
    cout << ">>> time cost: " << fixed << setprecision(PRECISION)
         << chrono::duration<double, milli>(diff).count()/LOOP_SIZE << "ms" << endl << endl;
}

int main(){
    ios_base::sync_with_stdio(false);

    random_device rd;
    default_random_engine e{rd()};
    uniform_int_distribution<int> u{0, 1000000};

    array<int, TEST_SIZE> test_array;

    for (int k = 0; k < TEST_SIZE; ++k)
        test_array[k] = u(e);

    time_test(); // 时间测试

    cout << endl << "--- 原始数组序列 ---" << endl << ">>>\t";
    print(test_array);
    cout << endl;

    insert_sort(test_array);
    bi_insert_sort(test_array);
    shell_sort(test_array);
    bubble_sort(test_array);
    select_sort(test_array);
    quick_sort(test_array);
    heap_sort(test_array);
    merge_sort(test_array);
    std_sort(test_array);
}
