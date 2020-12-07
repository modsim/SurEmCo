#include <ostream>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <numeric>
#include <vector>
#include <memory>
#include <iterator>

#include <iostream>

using namespace std;

/*
// Code for basic memory tracking

#include <new>
#include <cstring>

void *real_new(size_t size) {
    printf("new [%zu] ", size);
    void *ptr = malloc(size + sizeof(size_t));
    ((size_t*)ptr)[0] = size;
    ptr = ptr + sizeof(size_t);
    printf(" - %zu\n", ptr);
    return ptr;
}

void real_delete(void *ptr) {

    ptr -= sizeof(size_t);

    printf("delete %zu (%zu)", ptr + sizeof(size_t), ((size_t*)ptr)[0]);
    free(ptr);
}

void* operator new(size_t size) { return real_new(size); }
void* operator new[](size_t size) { return real_new(size); }
void* operator new(size_t size, const nothrow_t &) throw () { return real_new(size); }
void* operator new[](size_t size, const nothrow_t &) throw ()  { return real_new(size); }

void operator delete(void *ptr) { real_delete(ptr); }
void operator delete[](void *ptr) { real_delete(ptr); }

void operator delete(void *ptr, const nothrow_t &) throw () { real_delete(ptr); }
void operator delete[](void *ptr, const nothrow_t &) throw ()  { real_delete(ptr); }
*/

#undef CLI

#define WITH_KDTREE 1

#ifdef WITH_KDTREE
#include "nanoflann.hpp"
using namespace nanoflann;
#endif



template<typename T> struct dumb_set {
    vector<T> backing;

    typedef typename vector<T>::iterator iterator;

    inline iterator find(T value) {
        return std::find(backing.begin(), backing.end(), value);
    }

    inline void insert(T value) {
        backing.push_back(value);
    }

    inline void erase(T value) {
        iterator position = find(value);

        if(position != backing.end()) {
            if(*position != backing.back()) {
                swap(*position, backing.back());
            }
            backing.pop_back();
        }
    }

    inline size_t size() {
        return backing.size();
    }

    inline iterator begin() {
        return backing.begin();
    }

    inline iterator end() {
        return backing.end();
    }
};

template<typename FT> struct point2d {
    FT x;
    FT y;

    inline point2d() : x(0.0), y(0.0) {}
    inline point2d(FT x, FT y) : x(x), y(y) {}
    inline operator point2d<float>() const { return point2d<float>(x, y); }
    inline operator point2d<double>() const { return point2d<double>(x, y); }

    inline friend ostream& operator<<(ostream &os, const point2d& p) {
        return os << "(" << p.x << ", " << p.y << ")";
    };

    inline FT distance_square(const point2d& p) {
        FT xdelta = x - p.x, ydelta = y - p.y;

        return xdelta * xdelta + ydelta * ydelta;
    }
};

template<typename FT> struct point2dprec : point2d<FT> {
    FT precision;
};

template<typename FT, typename IT> struct dataset {

    struct emitter {
        point2dprec<FT> position;
        IT frame;

        size_t index;

        IT label;

        FT square_displacement;

        emitter() : position(), frame(0), index(0), label(0), square_displacement(0) {}

        inline friend ostream& operator<<(ostream &os, const emitter& e) {
            return os << "[" << e.position << ", " << e.frame << ", " << e.index << ", " << e.label << "]";
        };
    };



    typedef typename vector< emitter >::iterator emitter_iterator;

    struct possible_linking {
        emitter_iterator earlier;
        emitter_iterator current;

        FT distance_measure;

        inline possible_linking(emitter_iterator earlier, emitter_iterator current, FT distance_measure) :
            earlier(earlier), current(current), distance_measure(distance_measure) {};

        inline bool operator<(possible_linking& b) {
            return distance_measure < b.distance_measure;
        };
    };

    // list of all emitters
    vector< emitter > emitters;

    // list iterators for each frame, denoting the subset of emitters on that frame (for a sorted list of emitters)
    vector< pair< emitter_iterator, emitter_iterator > > frame_positions;

    // total count of frames
    size_t frames;


    #ifdef WITH_KDTREE
    struct kdtree_adaptor {
        pair< emitter_iterator, emitter_iterator > position;

        inline kdtree_adaptor(pair< emitter_iterator, emitter_iterator > pos) : position(pos) {};

        inline size_t kdtree_get_point_count() const { return position.second - position.first; }

        inline FT kdtree_distance(const FT *p1, const size_t idx, size_t) const {
            const FT d0 = p1[0] - operator[](idx)->position.x;
            const FT d1 = p1[1] - operator[](idx)->position.y;
            return d0 * d0 + d1 * d1;
        }

        inline FT kdtree_get_pt(const size_t idx, int dim) const {
		    if (dim==0) return operator[](idx)->position.x;
		    else return operator[](idx)->position.y;
		}

		template <class BBOX> bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

		inline emitter_iterator operator[](size_t idx) const {
		    assert((position.first + idx) < position.second);
		    return position.first + idx;
		}
    };

    typedef KDTreeSingleIndexAdaptor< L2_Adaptor<FT, kdtree_adaptor>, kdtree_adaptor, 2> kdtree;

    #endif

    inline void sort_by_frame() {
        sort(emitters.begin(), emitters.end(), [](emitter& a, emitter& b) { return a.frame < b.frame; });
    };

    inline void sort_by_index() {
        sort(emitters.begin(), emitters.end(), [](emitter& a, emitter& b) { return a.index < b.index; });
    };



    inline void prepare() {
        sort_by_frame();

        frames = emitters.back().frame;

        auto the_end = pair<emitter_iterator, emitter_iterator>(emitters.end(), emitters.end());

        frame_positions.resize(frames + 1, the_end);

        for(emitter_iterator em_it = emitters.begin(); em_it != emitters.end(); em_it++ ) {
            if(frame_positions[em_it->frame].first == emitters.end()) {
                frame_positions[em_it->frame].first = em_it;
            }
            frame_positions[em_it->frame].second = em_it + 1;
        }
    };


    enum search_mode_type {
        SEARCH_MODE_BRUTE_FORCE = 0,
        #ifdef WITH_KDTREE
        SEARCH_MODE_KD_TREE = 1
        #endif
    };

    enum tracking_mode_type {
        TRACKING_MOVING = 0,
        TRACKING_STATIC = 1
    };

    typedef dumb_set< emitter_iterator > emitter_set;

    inline void link(FT max_distance = 2.0, IT memory = 0, search_mode_type search_mode = SEARCH_MODE_BRUTE_FORCE, tracking_mode_type tracking_mode = TRACKING_MOVING) {
        memory += 1; // we always look back at least one frame, otherwise we wouldn't track anything

        // label 0 is reserved for unassigned
        IT label = 1;

        FT max_distance_square = max_distance * max_distance;


        #ifdef WITH_KDTREE

        vector<kdtree_adaptor> frame_adaptors;
        vector<shared_ptr<kdtree>> frame_trees;

        frame_adaptors.reserve(frame_positions.size());
        frame_trees.reserve(frame_positions.size());

        if(search_mode == SEARCH_MODE_KD_TREE) {
            for(auto& position: frame_positions) {
                auto adaptor = kdtree_adaptor(position);
                auto tree = shared_ptr<kdtree>(new kdtree(2, adaptor, KDTreeSingleIndexAdaptorParams(10)));
                tree->buildIndex();

                frame_adaptors.push_back(adaptor);
                frame_trees.push_back(tree);
            }
        }
        #endif


        vector<point2d<FT> > mean_points;
        vector<FT> mean_precision;
        vector<size_t> mean_counts;

        set<emitter_iterator> linked;

        if(tracking_mode == TRACKING_STATIC) {
            // if we're using the static mode,
            // we resize
            mean_points.resize(emitters.size()+1);
            mean_precision.resize(emitters.size()+1);
            mean_counts.resize(emitters.size()+1);
        }

        for(size_t current_frame = 0; current_frame <= frames; current_frame++) {
            emitter_set current_emitters_to_link;
            emitter_set earlier_emitters_to_link;
            vector<possible_linking> possibilities;

            for(auto current_emitter = frame_positions[current_frame].first; current_emitter != frame_positions[current_frame].second; current_emitter++) {
                current_emitters_to_link.insert(current_emitter);

                // we have an emitter 'current_emitter' ... now look back (the frames) if we found something similar

                size_t earliest_frame = (current_frame <= memory) ? 0 : current_frame - memory;

                for(size_t earlier_frame = current_frame - 1; (earlier_frame >= earliest_frame) && (earlier_frame < current_frame); earlier_frame--) {

                    // a very basic if in a hot loop,
                    // I don't like this, but it's probably for now the easiest way to have some configurable options
                    if(search_mode == SEARCH_MODE_BRUTE_FORCE) {

                        for(auto earlier_emitter = frame_positions[earlier_frame].first; earlier_emitter != frame_positions[earlier_frame].second; earlier_emitter++) {

                            //if(tracking_mode == TRACKING_MOVING)
                            point2d<FT>& position_to_compare = earlier_emitter->position;

                            if(tracking_mode == TRACKING_STATIC) {
                                position_to_compare = mean_points[earlier_emitter->label];
                                FT precision = mean_precision[earlier_emitter->label];
                                max_distance_square = precision * precision;
                            }

                            FT distance_square = current_emitter->position.distance_square(position_to_compare);

                            // max_distance_square is either constant

                            if(distance_square > max_distance_square)
                                continue;

                            earlier_emitters_to_link.insert(earlier_emitter);

                            possibilities.push_back(possible_linking(earlier_emitter, current_emitter, distance_square));

                        }

                    } // else
                    #ifdef WITH_KDTREE
                    if(search_mode == SEARCH_MODE_KD_TREE) {

                            vector<pair<size_t, FT> > matches;
                            matches.reserve(512); // have enough

                            SearchParams params;

                            FT factor = 1.0;

                            if(tracking_mode == TRACKING_STATIC) {
                                factor = 1.0;
                            }

                            point2d<FT> search = current_emitter->position;

                            auto& tree = frame_trees[earlier_frame];
                            auto adaptor = frame_adaptors[earlier_frame];



                            size_t count = tree->radiusSearch(&search.x, factor*max_distance_square + numeric_limits<FT>::epsilon(), matches, params);
                            for(size_t i = 0; i < count; i++) {
                                auto& match = matches[i];

                                auto earlier_emitter = adaptor[match.first];

                                //FT local_max_distance_square = max_distance_square;

                                if(tracking_mode == TRACKING_STATIC) {
                                    FT distance = current_emitter->position.distance_square(mean_points[earlier_emitter->label]);
                                    FT precision = mean_precision[earlier_emitter->label];
                                    if(distance > (precision * precision)) {
                                        continue;
                                    }
                                }

                                earlier_emitters_to_link.insert(earlier_emitter);
                                possibilities.push_back(possible_linking(earlier_emitter, current_emitter, match.second));

                            }

                        }
                    }
                    #endif
                }



            sort(possibilities.begin(), possibilities.end());

            for(auto& p: possibilities) {
                if(current_emitters_to_link.size() == 0)
                    break;

                if(current_emitters_to_link.find(p.current) == current_emitters_to_link.end())
                    continue;

                if(earlier_emitters_to_link.find(p.earlier) == earlier_emitters_to_link.end())
                    continue;

                if(linked.find(p.earlier) != linked.end())
                    continue;

                IT new_label = p.earlier->label;

                p.current->label = new_label;
                p.current->square_displacement = p.earlier->square_displacement + p.distance_measure;

                current_emitters_to_link.erase(p.current);
                earlier_emitters_to_link.erase(p.earlier);

                linked.insert(p.earlier);

                if(tracking_mode == TRACKING_STATIC) {
                    IT count_so_far = mean_counts[new_label];
                    IT new_count = count_so_far + 1;

                    mean_points[new_label] = point2d<FT>(
                        (mean_points[new_label].x * count_so_far + p.current->position.x) / new_count,
                        (mean_points[new_label].y * count_so_far + p.current->position.y) / new_count
                    );

                    mean_precision[new_label] = (mean_precision[new_label] * count_so_far + p.current->position.precision) / new_count;

                    mean_counts[new_label] = new_count;
                }
            }

            for(auto& em_it : current_emitters_to_link) {
                IT new_label = label++;
                em_it->label = new_label;

                if(tracking_mode == TRACKING_STATIC) {
                    mean_points[new_label] = em_it->position;
                    mean_precision[new_label] = em_it->position.precision;
                    mean_counts[new_label] = 1;
                }
            }

        }

        for(auto& e : emitters) {
            if(e.label == 0) {
                e.label = label++;
            }
        }

    }

};


template<typename FT> ostream& print_vector(ostream &os, const vector<FT>& e) {
    os << "vector [" << endl;
    for(size_t i = 0; i < e.size(); i++) {
        os << "\t" << e[i] << endl;
    }
    os << "]" << endl;
    return os;
};

typedef dataset<double, size_t> dataset_type;

extern "C" const char *getBuildDate() {
    return __DATE__ " " __TIME__;
}

extern "C" void track(dataset_type::emitter *input_data, size_t count, float max_distance, int memory, int mode, int strategy) {

    dataset_type data;

    data.emitters.resize(count);

    copy(input_data, input_data + count, data.emitters.begin());
    data.prepare();

    data.link(max_distance, memory, (dataset_type::search_mode_type)strategy, (dataset_type::tracking_mode_type)mode);

    data.sort_by_index();

    copy(data.emitters.begin(), data.emitters.end(), input_data);
};


extern "C" float msd(dataset_type::emitter *input_data, size_t count, float MICRON_PER_PIXEL, float FRAMES_PER_SECOND) {
    double MICRON_PER_PIXEL_SQ = MICRON_PER_PIXEL * MICRON_PER_PIXEL;

    dataset_type data;

    data.emitters.resize(count);

    copy(input_data, input_data + count, data.emitters.begin());
    sort(data.emitters.begin(), data.emitters.end(), [](dataset_type::emitter& a, dataset_type::emitter& b) { return a.label < b.label || (a.label == b.label && a.frame < b.frame); });


    vector<double> square_displacements;
    square_displacements.reserve(data.emitters.size());
    dataset_type::emitter& first = data.emitters[0];

    double dimensionality = 2.0;
    double two_times_dimensionality = 2.0 * dimensionality;

    for(auto& e: data.emitters) {
        if(first.index == e.index || first.label != e.label || e.square_displacement == 0.0) {
            first = e;
            continue;
        }

        double sqd = e.square_displacement * MICRON_PER_PIXEL_SQ;
        double t = (e.frame - first.frame) / FRAMES_PER_SECOND;


        square_displacements.push_back(sqd/t);
    }

    double D = (accumulate(square_displacements.begin(), square_displacements.end(), 0.0) / square_displacements.size()) / two_times_dimensionality;

    return D;
};
