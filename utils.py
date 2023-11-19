import os
import cv2
import math
import time
import cupy as cp
import numpy as np
from datetime import date
from ultralytics import YOLO
from screeninfo import get_monitors


def lookup_table(tamanhoFrame):
    rpixel = Vpn = int(tamanhoFrame)
    Hpn = int(Vpn*2*cp.pi)
    upn, vpn = cp.meshgrid(cp.arange(0, Hpn, 1), cp.arange(0, Vpn, 1))
    uimg = (((vpn*rpixel)/Vpn)*cp.cos((upn*2*cp.pi)/Hpn) + rpixel).astype(int)
    vimg = (-(((vpn*rpixel)/Vpn)*cp.sin((upn*2*cp.pi)/Hpn) - rpixel)).astype(int)
    return uimg, vimg


def lookup_table_numpy(tamanhoFrame):
    rpixel = Vpn = int(tamanhoFrame)
    Hpn = int(Vpn*2*np.pi)
    upn, vpn = np.meshgrid(np.arange(0, Hpn, 1), np.arange(0, Vpn, 1))
    uimg = (((vpn*rpixel)/Vpn)*np.cos((upn*2*np.pi)/Hpn) + rpixel).astype(int)
    vimg = (-(((vpn*rpixel)/Vpn)*np.sin((upn*2*np.pi)/Hpn) - rpixel)).astype(int)
    return uimg, vimg


def lookup_table_cupy(tamanhoFrame):
    rpixel = Vpn = int(tamanhoFrame)
    Hpn = int(Vpn*2*cp.pi)
    upn, vpn = cp.meshgrid(cp.arange(0, Hpn, 1), cp.arange(0, Vpn, 1))
    uimg = (((vpn*rpixel)/Vpn)*cp.cos((upn*2*cp.pi)/Hpn) + rpixel).astype(int)
    vimg = (-(((vpn*rpixel)/Vpn)*cp.sin((upn*2*cp.pi)/Hpn) - rpixel)).astype(int)
    return uimg, vimg


def cria_pasta(nome_pasta):
    pasta = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), nome_pasta)
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    return pasta


def obter_nome_unico(caminho_completo):
    nome, extensao = os.path.splitext(caminho_completo)
    contador = 1
    caminho_completo = f"{nome}_00{extensao}"
    while os.path.exists(caminho_completo):
        if contador < 10:
            caminho_completo = f"{nome}_0{contador}{extensao}"
        else:
            caminho_completo = f"{nome}_{contador}{extensao}"
        contador += 1
    return caminho_completo


def configurar_camera(camera):
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, largura_camera)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, altura_camera)
    camera.set(cv2.CAP_PROP_FPS, fps_camera)


stream = cp.cuda.Stream()
device = cp.cuda.Device(0)

monitor = get_monitors()

tempo_captura_frame = 15

ajusteCentro = (-10, 22)  # (0, 20)
corteInf = 90
corteSup = 15
hoje = date.today().strftime("%Y-%m-%d")[2:]

largura_camera, altura_camera = 1280, 720
fps_camera = 200.0

phi_max = 15
phi_min = 20

classes_yolo = []
for i in range(80):
    if i <= 36 and i not in [4, 6, 19]:
        classes_yolo.append(i)
# print(classes_yolo)

conf_yolo = 0.35
maxima_det = 20
