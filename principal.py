# -*- coding: utf-8 -*-
# Ana Lucia Hernandez
# 17138
# Graficas por Computadora 

'''

        ---- DISCLAIMER ----
    Los archivos de base_noise.py, constants.py, perlin.py y hammersley.py - que son necesarios para crear ruido 
    en el renderizado - fueron obtenidos de: https://github.com/mmchugh/pynoise
    Créditos al autor, Michael McgHugh.
'''

import shaders as s

s.bm = s.Bitmap(500,500)
s.Estrellas()
#                      translate    scale      rotate      eye       up      center     luz
s.bm.load('esfera.obj', (0,0,0), (0.8,0.8,0.8), (0,0,0), (0,0,5), (0,1,0), (0,0,0), (1.2,0.1,0.7))
s.bm.load('ring.obj', (0,0,0), (0.6,0.9,0.9), (0,0,0), (0,0,5), (0,1,0), (0,0,0), (0.7,0.1, 1.2))

s.glFinish("Jupiter")