#include <ostream>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <vector>
#include <cstring>

#undef CLI

using namespace std;


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
            swap(*position, backing.back());
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

        emitter() : position(), frame(0), index(0), label(0) {}

        inline friend ostream& operator<<(ostream &os, const emitter& e) {
            return os << "[" << e.position << ", " << e.frame << ", " << e.index << ", " << e.label << "]";
        };

        inline static bool frame_comparator(emitter a, emitter b) {
            return a.frame < b.frame;
        };

        inline static bool index_comparator(emitter a, emitter b) {
            return a.index < b.index;
        };
    };

    struct possible_linking {
        size_t index_earlier;
        size_t index_current;

        FT distance_measure;

        inline bool operator<(possible_linking& b) {
            return distance_measure < b.distance_measure;
        };
    };




    vector< emitter > emitters;

    vector< bool > is_linked;

    vector< size_t > frame_positions;

    size_t frames;

    size_t NO_EMITTERS;


    inline void sort_by_frame() {
        sort(emitters.begin(), emitters.end(), emitter::frame_comparator);
    };

    inline void sort_by_index() {
        sort(emitters.begin(), emitters.end(), emitter::index_comparator);
    };



    inline void prepare() {
        sort_by_frame();

        size_t frame = 0;

        for(size_t i = 0; i < emitters.size(); i++) {
            frames = emitters[i].frame;
        }

        frame_positions.resize(frames + 1);

        NO_EMITTERS = 0;
        NO_EMITTERS --;

        fill(frame_positions.begin(), frame_positions.end(), NO_EMITTERS);

        frame_positions[0] = 0;


        for(size_t i = 0; i < emitters.size(); i++) {
            frame_positions.at(emitters[i].frame + 1) = i + 1;
        }

        frame_positions[frames + 1] = emitters.size();

        is_linked.resize(emitters.size());
    }

    enum search_mode_type {
        BRUTE_FORCE = 0
    };

    enum tracking_mode_type {
        TRACKING_MOVING = 0,
        TRACKING_STATIC = 1
    };

    typedef dumb_set<size_t> emitter_set;
    //typedef set<size_t> emitter_set;

    inline void link(FT max_distance = 2.0, IT memory = 1, search_mode_type search_mode = BRUTE_FORCE, tracking_mode_type tracking_mode = TRACKING_MOVING) {

        // label 0 is reserved for unassigned
        IT label = 1;
        FT max_distance_square = max_distance * max_distance;

        vector<point2d<FT> > mean_points;
        vector<FT> mean_precision;
        vector<size_t> mean_counts;


        if(tracking_mode == TRACKING_STATIC) {
            // if we're using the static mode,
            // we resize
            mean_points.resize(emitters.size()+1);
            mean_precision.resize(emitters.size()+1);
            mean_counts.resize(emitters.size()+1);
        }

        for(size_t current_frame = 0; current_frame < frames; current_frame++) {

            size_t current_frame_min = frame_positions[current_frame], current_frame_max = frame_positions[current_frame+1];

            if(current_frame_min == NO_EMITTERS)
                continue;

            emitter_set current_emitters_to_link;
            emitter_set earlier_emitters_to_link;
            vector<possible_linking> possibilities;

            for(size_t current_emitter_e = current_frame_min; current_emitter_e < current_frame_max; current_emitter_e++) {
                emitter& current_emitter = emitters[current_emitter_e];

                current_emitters_to_link.insert(current_emitter_e);

                // we have an emitter 'current_emitter' ... now look back (the frames) if we found something similar

                size_t earliest_frame = (current_frame <= memory) ? 0 : current_frame - memory;

                if(current_frame > 0) {
                    // unsigned wraparound!
                    for(size_t earlier_frame = current_frame - 1; (earlier_frame >= earliest_frame) && (earlier_frame < current_frame); earlier_frame--) {
                        size_t earlier_frame_min = frame_positions[earlier_frame], earlier_frame_max = frame_positions[earlier_frame+1];
                        for(size_t earlier_emitter_e = earlier_frame_min; earlier_emitter_e < earlier_frame_max; earlier_emitter_e++) {
                            emitter& earlier_emitter = emitters[earlier_emitter_e];

                            FT distance_square;

                            if(tracking_mode == TRACKING_MOVING) {
                                distance_square = current_emitter.position.distance_square(earlier_emitter.position);

                                if(distance_square > max_distance_square)
                                    continue;

                            } else if(tracking_mode == TRACKING_STATIC) {
                                distance_square = current_emitter.position.distance_square(mean_points[earlier_emitter.label]);

                                FT prec_square = mean_precision[earlier_emitter.label];

                                prec_square = prec_square * prec_square;

                                if(distance_square > prec_square)
                                    continue;
                            }



                            //cout << "hit" << endl;

                            earlier_emitters_to_link.insert(earlier_emitter_e);

                            possible_linking pl;

                            pl.index_earlier = earlier_emitter_e;
                            pl.index_current = current_emitter_e;

                            pl.distance_measure = distance_square;

                            possibilities.push_back(pl);

                        }
                    }
                }
            }

            sort(possibilities.begin(), possibilities.end());

            for(size_t i = 0; i < possibilities.size(); i++) {
                if(current_emitters_to_link.size() == 0)
                    break;

                possible_linking& p = possibilities[i];

                if(current_emitters_to_link.find(p.index_current) == current_emitters_to_link.end())
                    continue;

                if(earlier_emitters_to_link.find(p.index_earlier) == earlier_emitters_to_link.end())
                    continue;

                if(is_linked[p.index_earlier])
                    continue;

                IT new_label = emitters[p.index_earlier].label;

                emitters[p.index_current].label = new_label;

                current_emitters_to_link.erase(p.index_current);
                earlier_emitters_to_link.erase(p.index_earlier);

                is_linked[p.index_earlier] = true;

                if(tracking_mode == TRACKING_STATIC) {
                    IT count_so_far = mean_counts[new_label];
                    IT new_count = count_so_far + 1;

                    mean_points[new_label] = point2d<FT>(
                        (mean_points[new_label].x * count_so_far + emitters[p.index_current].position.x) / new_count,
                        (mean_points[new_label].y * count_so_far + emitters[p.index_current].position.y) / new_count
                    );

                    mean_precision[new_label] = (mean_precision[new_label] * count_so_far + emitters[p.index_current].position.precision) / new_count;

                    mean_counts[new_label] = new_count;
                }
            }

            for(emitter_set::iterator it = current_emitters_to_link.begin(); it != current_emitters_to_link.end(); it++) {
                IT new_label = label++;
                emitters[*it].label = new_label;

                if(tracking_mode == TRACKING_STATIC) {
                    mean_points[new_label] = emitters[*it].position;
                    mean_precision[new_label] = emitters[*it].position.precision;
                    mean_counts[new_label] = 1;
                }
            }

        }
    }

    void fancy_print(ostream &os) {
        for(size_t f = 0; f < frames; f++) {
            size_t min = frame_positions[f], max = frame_positions[f+1];
            os << "Frame " << f << " with " << (max-min) << " emitters [" << endl;

            for(size_t e = min; e < max; e++) {
                os << "\t" << emitters[e] << endl;
            }

            os << "]" << endl;
        }

    };


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

extern "C" void track(dataset_type::emitter *input_data, size_t count, float max_distance, int memory, int mode) {

    dataset_type data;

    data.emitters.resize(count+1);

    memcpy(data.emitters.data(), input_data, count * sizeof(dataset_type::emitter));

    data.prepare();

    data.link(max_distance, memory, dataset_type::BRUTE_FORCE, (dataset_type::tracking_mode_type)mode);

    data.sort_by_index();

    memcpy(input_data, data.emitters.data(), count * sizeof(dataset_type::emitter));

};
