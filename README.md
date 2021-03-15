# SurEmCo

The *Su*per*r*esolution *Em*itter *Co*unter, as used to perform some analysis steps of the following publication:

```
Growth-rate dependency of translation rate and active ribosome pool in Corynebacterium glutamicum differ from that of Escherichia coli
Matamouros S., Gensch T., Cerff M., Sachs C.C., Abdoullahzadeh I., Hendriks J., Horst L., Tenhaef N., Noack, S., Graf M., Takors R., Nöh K. and Bott M.
In preparation.
```

Preliminary - the final citation will be updated upon publication.

The analysis routines are tailored to the use case of the publication. It may be necessary to perform adaptations for other uses.
Please cite our publication in case you use SurEmCo. 

# Installation

Compilation of the tracker has only been tested with Ubuntu Linux (among others, the `g++ g++-mingw-w64-x86-64` packages might be needed), where it is cross-compiled for Windows as well.

To just use SurEmCo, it is recommended to install it via Anaconda (the additional `conda-forge` channel needs to be activated beforehand (`conda config --add channels conda-forge`):
```
> conda install -yc modsim suremco
```

It can then be started via
```
> python -m suremco
# or with filenames
> python -m suremco dia_image.png snsmil_output.txt
```

# License

BSD
