// hnsw_wrapper.cc
#include <cstring>
#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
#include <thread>
#include <atomic>
#include <vector>


// Error handling helper: copies a string into malloc'd memory for return to Go via char **err.
static char* copyErrorString(const char* msg) {
    if (!msg) return nullptr;
    size_t len = strlen(msg) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, msg, len);
    return copy;
}

// Resolves the effective thread count: clamps to at least 1, falls back to hardware_concurrency.
static int resolveThreadCount(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 1;
    }
    return num_threads;
}

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1)
    {
        for (size_t id = start; id < end; id++)
        {
            fn(id, 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.push_back(std::thread([&, threadId]
                                          {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        if (lastException)
        {
            std::rethrow_exception(lastException);
        }
    }
}

HnswIndex *newIndex(spaceType space_type, const int dim, size_t max_elements, int M, int ef_construction, int rand_seed, int allow_replace_deleted, char **err)
{
    HnswIndex *index = nullptr;
    hnswlib::SpaceInterface<float> *space = nullptr;

    try {
        index = new HnswIndex;
        bool normalize = false;

        if (space_type == l2)
        {
            space = new hnswlib::L2Space(dim);
        }
        else if (space_type == ip)
        {
            space = new hnswlib::InnerProductSpace(dim);
        }
        else if (space_type == cosine)
        {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        }
        else
        {
            delete index;
            if (err) *err = copyErrorString("Space name must be one of l2, ip, or cosine.");
            return nullptr;
        }

        hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction, rand_seed, static_cast<bool>(allow_replace_deleted));

        index->hnsw = (void *)appr_alg;
        index->dim = dim;
        index->normalize = normalize;
        index->space = (void *)space;
        index->space_type = space_type;
        return index;
    } catch (const std::exception& e) {
        delete space;
        delete index;
        if (err) *err = copyErrorString(e.what());
        return nullptr;
    }
}

// set efConstruction value.
void setEf(HnswIndex *index, size_t ef)
{
    ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->ef_ = ef;
}

// Returns index file size in size_t via out-param. Returns 0 on success, 1 on error.
int indexFileSize(HnswIndex *index, size_t *result, char **err)
{
    try {
        *result = ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->indexFileSize();
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

// Save index to a file.
int saveIndex(HnswIndex *index, char *location, char **err)
{
    try {
        ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->saveIndex(location);
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

HnswIndex *loadIndex(char *location, spaceType space_type, int dim, size_t max_elements, int allow_replace_deleted, char **err)
{
    HnswIndex *index = nullptr;
    hnswlib::SpaceInterface<float> *space = nullptr;

    try {
        index = new HnswIndex;
        bool normalize = false;

        if (space_type == l2)
        {
            space = new hnswlib::L2Space(dim);
        }
        else if (space_type == ip)
        {
            space = new hnswlib::InnerProductSpace(dim);
        }
        else if (space_type == cosine)
        {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        }
        else
        {
            delete index;
            if (err) *err = copyErrorString("Space name must be one of l2, ip, or cosine.");
            return nullptr;
        }

        hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, location, false, max_elements, static_cast<bool>(allow_replace_deleted));

        index->hnsw = (void *)appr_alg;
        index->dim = dim;
        index->normalize = normalize;
        index->space = (void *)space;
        index->space_type = space_type;
        return index;
    } catch (const std::exception& e) {
        delete space;
        delete index;
        if (err) *err = copyErrorString(e.what());
        return nullptr;
    }
}

static void normalize_vector(int dim, const float *data, float *norm_array)
{
    float norm = 0.0f;
    for (int i = 0; i < dim; i++)
        norm += data[i] * data[i];
    norm = 1.0f / (sqrtf(norm) + 1e-30f);
    for (int i = 0; i < dim; i++)
        norm_array[i] = data[i] * norm;
}

int addPoints(HnswIndex *index, const float *flat_vectors, int rows, size_t *labels, int num_threads, int replace_deleted, char **err)
{
    try {
        num_threads = resolveThreadCount(num_threads);

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4)
        {
            num_threads = 1;
        }

        int d = index->dim;

        if (index->normalize == false) {
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t id = *(labels + row);
                ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->addPoint(flat_vectors + row * d, id, static_cast<bool>(replace_deleted));
            });
            return 0;
        }

        std::vector<float> norm_array(num_threads * d);
        ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId){
            size_t start_idx = threadId * d;
            normalize_vector(d, flat_vectors + row * d, norm_array.data() + start_idx);

            size_t id = *(labels + row);
            ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->addPoint((void*)(norm_array.data() + start_idx), id, static_cast<bool>(replace_deleted));
            });

    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }

    return 0;
}

int markDeleted(HnswIndex *index, size_t label, char **err)
{
    try {
        ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->markDelete(label);
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

int unmarkDeleted(HnswIndex *index, size_t label, char **err)
{
    try {
        ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->unmarkDelete(label);
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

int resizeIndex(HnswIndex *index, size_t new_size, char **err)
{
    try {
        ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->resizeIndex(new_size);
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

size_t getMaxElements(HnswIndex *index)
{
    return ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->max_elements_;
}

size_t getCurrentCount(HnswIndex *index)
{
    return ((hnswlib::HierarchicalNSW<float> *)(index->hnsw))->cur_element_count;
}

SearchResult *searchKnn(HnswIndex *index, const float *flat_vectors, int rows, int k, int num_threads, char **err)
{
    SearchResult *searchResult = nullptr;
    try {
        num_threads = resolveThreadCount(num_threads);

        // avoid using threads when the number of searches is small:
        if (rows <= num_threads * 4)
        {
            num_threads = 1;
        }

        int d = index->dim;

        searchResult = new SearchResult;
        searchResult->label = nullptr;
        searchResult->dist = nullptr;
        searchResult->label = new hnswlib::labeltype[rows * k];
        searchResult->dist = new float[rows * k];

        if (index->normalize == false) {
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                    ((hnswlib::HierarchicalNSW<float> *)index->hnsw)->searchKnn(flat_vectors + row * d, k, nullptr);

                if (result.size() != (size_t)k)
                    throw std::runtime_error("Cannot return the results in a contiguous 2D array. Probably ef or M is too small");

                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    *(searchResult->dist + row * k + i) = result_tuple.first;
                    *(searchResult->label + row * k + i) = result_tuple.second;
                    result.pop();
                }
            });

        } else {
            std::vector<float> norm_array(num_threads * d);
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t start_idx = threadId * d;
                normalize_vector(d, flat_vectors + row * d, norm_array.data() + start_idx);

                std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                    ((hnswlib::HierarchicalNSW<float> *)index->hnsw)->searchKnn((void*)(norm_array.data() + start_idx), k, nullptr);

                if (result.size() != (size_t)k)
                    throw std::runtime_error("Cannot return the results in a contiguous 2D array. Probably ef or M is too small");

                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    *(searchResult->dist + row * k + i) = result_tuple.first;
                    *(searchResult->label + row * k + i) = result_tuple.second;
                    result.pop();
                }
            });
        }

        return searchResult;
    } catch (const std::exception& e) {
        if (searchResult) {
            delete[] searchResult->label;
            delete[] searchResult->dist;
            delete searchResult;
        }
        if (err) *err = copyErrorString(e.what());
        return nullptr;
    }
}

int getAllowReplaceDeleted(HnswIndex *index) {
   return ((hnswlib::HierarchicalNSW<float> *)index->hnsw)->allow_replace_deleted_;
}

int getDataByLabel(HnswIndex *index, const size_t label, float* data, char **err) {
    try {
        auto vec = ((hnswlib::HierarchicalNSW<float> *)index->hnsw)->getDataByLabel<float>(label);
        memcpy(data, vec.data(), sizeof(float) * index->dim);
        return 0;
    } catch (const std::exception& e) {
        if (err) *err = copyErrorString(e.what());
        return 1;
    }
}

void freeHNSW(HnswIndex *index)
{
    if (!index) return;

    try {
        if (index->hnsw) {
            delete (hnswlib::HierarchicalNSW<float> *)index->hnsw;
        }

        if (index->space_type == l2)
        {
            delete (hnswlib::L2Space *)(index->space);
        }
        else if (index->space_type == ip || index->space_type == cosine)
        {
            delete (hnswlib::InnerProductSpace *)(index->space);
        }

        delete index;
    } catch (...) {
        // Best effort cleanup -- do not throw through extern "C" boundary
    }
}

void freeResult(SearchResult *result)
{
    if (!result) return;
    delete[] result->label;
    delete[] result->dist;
    delete result;
}
