// hnsw_wrapper.h
#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *HNSW;
    typedef void *HnswSpace;
    typedef enum {
        l2, ip, cosine
    } spaceType;

    // The index wrapper with some needed properties if initialized index.
    typedef struct
    {
        HNSW hnsw;
        HnswSpace space;
        spaceType space_type;
        int dim;
        int normalize;
    } HnswIndex;

    // SearchResult holds the multi-vector search result. label and dist are flatted 2d vectors.
    typedef struct
    {
        size_t *label;
        float *dist;
    } SearchResult;

    HnswIndex *newIndex(spaceType space_type, const int dim, size_t max_elements, int M, int ef_construction, int rand_seed, int allow_replace_deleted, char **err);
    void setEf(HnswIndex *index, size_t ef);
    int indexFileSize(HnswIndex *index, size_t *result, char **err);
    int saveIndex(HnswIndex *index, char *location, char **err);
    HnswIndex *loadIndex(char *location, spaceType space_type, int dim, size_t max_elements, int allow_replace_deleted, char **err);

    int addPoints(HnswIndex *index, const float *vectors, int rows, size_t *labels, int num_threads, int replace_deleted, char **err);
    int markDeleted(HnswIndex *index, size_t label, char **err);
    int unmarkDeleted(HnswIndex *index, size_t label, char **err);
    int resizeIndex(HnswIndex *index, size_t new_size, char **err);
    size_t getMaxElements(HnswIndex *index);
    size_t getCurrentCount(HnswIndex *index);
    int getAllowReplaceDeleted(HnswIndex *index);
    SearchResult *searchKnn(HnswIndex *index, const float *flat_vectors, int rows, int k, int num_threads, char **err);

    int getDataByLabel(HnswIndex *index, const size_t label, float *data, char **err);
    void freeHNSW(HnswIndex *index);
    void freeResult(SearchResult *result);

#ifdef __cplusplus
}
#endif
