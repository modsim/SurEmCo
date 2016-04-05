g++ -g -std=c++11 -O3 tracker.cpp -o superresolution/_tracker.so -fPIC -shared && \
x86_64-w64-mingw32-g++ -g -std=c++11 -shared -static-libgcc -static-libstdc++ /usr/x86_64-w64-mingw32/lib/libwinpthread.a -o superresolution/_tracker.dll tracker.cpp -O3

