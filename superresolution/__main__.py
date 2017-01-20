# encoding: utf-8
# Copyright (C) 2015-2017 Christian C. Sachs, Forschungszentrum Jülich

from .app import SuperresolutionTracking

#####
"""
512 x 512 ... 0.065 um per pixel
2560 x 2560 ... 0.013 um per pixel


Average ribosome diffusion rate: 0.04 +- 0.01 square micrometer per second
assumption: all fluorescent ribosomal proteins are bound in ribosomes

exposure/timing:
50ms exposure + 36ms delay
(exposure every 86ms)

"""


#####

def main():
    SuperresolutionTracking.run()

if __name__ == '__main__':
    main()
