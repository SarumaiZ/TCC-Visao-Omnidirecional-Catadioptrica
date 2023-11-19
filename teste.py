from utils import *

# import cv2

# img_orig = cv2.imread("imagemEspelhoHiperbolico.png")
# img_pan = cv2.imread("resultados\imagem_Transformada.jpg")
# print(f"Tamanho imagem original: {img_orig.shape[0]} x {img_orig.shape[1]}\nQuantidade de pixels: {img_orig.shape[0]*img_orig.shape[1]}")
# print(f"Tamanho imagem panoramica: {img_pan.shape[0]} x {img_pan.shape[1]}\nQuantidade de pixels: {img_pan.shape[0]*img_pan.shape[1]}")

# from datetime import date

# hoje = date.today().strftime("%Y-%m-%d")[2:]
# print(hoje)

# import cv2
# frames = []
# cam = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# # out = cv2.VideoWriter("video.avi", fourcc, 30, (640,480))
# while True:
#     ret, frame = cam.read()
#     cv2.imshow("Video", frame)
#     # out.write(frame)
#     frames.append(frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cam.release()
# # out.release()
# cv2.destroyAllWindows()

# cv2.imshow("frame", frames[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(frames[0].shape)

# h, w, _ = frames[0].shape
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter("video.avi", fourcc, 30, (w, h))
# for frame in frames:
#     writer.write(frame)
# writer.release()

# import os
# from datetime import date
# import cv2
# hoje = date.today().strftime("%Y-%m-%d")[2:]

# def obter_nome_unico(fps):
#     pasta_frames = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
#     if not os.path.exists(pasta_frames):
#         os.makedirs(pasta_frames)
#     caminho_completo_frame = os.path.join(pasta_frames, "0_" + hoje + "-teste-" + str(fps//10) + ".jpg")
#     contador = 1
#     while os.path.exists(caminho_completo_frame):
#         caminho_completo_frame = os.path.join(pasta_frames, f"{contador}_{hoje}-teste-{str(fps//10)}.jpg")
#         contador += 1
#     return caminho_completo_frame

# imagem = cv2.imread("imagens\imagemEspelhoHiperbolico.png")
# fps = 0
# for i in range(100):
#     if (fps%10 == 0):
#         caminho_completo_frame = obter_nome_unico(fps)
#         cv2.imwrite(caminho_completo_frame, imagem)
#     fps+=1

# from screeninfo import get_monitors

# # Obtém a lista de monitores disponíveis
# monitor = get_monitors()

# img shape 1 após retificada: (421, 3304, 3)
# img shape pos resolucao: (194, 1526, 3)
# resolucaoFinal = 0.462

# import cv2

# pan_grande = cv2.imread(r"resultados\23-08-13-imagem_Transformada_1.jpg")
# pan_pequena = cv2.imread(r"resultados\23-08-13-imagem_Transformada_2.jpg")

# print(f"Tamanho do monitor: {monitor[0].height} x {monitor[0].width}")
# print(f"Tamanho da panoramica grande: {pan_grande.shape}")
# print(f"Tamanho da panoramica pequena: {pan_pequena.shape}")
# print(f"Razaão de diminuição: {round(monitor[0].width/pan_pequena.shape[1]-0.119, 3)}")

# import time
# from ultralytics import YOLO
# import cv2
# import cupy as cp
# from screeninfo import get_monitors

# monitor = get_monitors()

# ajusteCentro = (0, -14)
# corteInf = 90
# corteSup = 15

# def retifica_imagem(frame):
#     img = cp.copy(frame)
#     resolucao = (frame.shape[1], frame.shape[0])
#     centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
#     tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))

#     # Calculos de parametros da transformação
#     rpixel = Vpn = int(tamanhoFrame)
#     Hpn = int(Vpn*2*cp.pi) # Comprimento da imagem nova .astype(cp.uint8)

#     # Criação das matrizes para o calculo
#     uvect = cp.arange(0, Hpn, 1) #Vetor de coordenadas em x
#     vvect = cp.arange(0, Vpn, 1) #Vetor de coordenadas em y
#     upn, vpn = cp.meshgrid(uvect, vvect) #matriz de coordenadasd vetorizadas

#     # Transformação de coordenadas
#     uimg = (((vpn*rpixel)/Vpn)*cp.cos((upn*2*cp.pi)/Hpn) + rpixel).astype(int)
#     vimg = (-(((vpn*rpixel)/Vpn)*cp.sin((upn*2*cp.pi)/Hpn) - rpixel)).astype(int)
#     img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame, centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
#     # img = (img[vimg.get(), uimg.get()]/255)
#     img = img[vimg.get(), uimg.get()]
#     img = cp.flip(img, 0)
#     img = img[corteSup:-corteInf, :]
#     resolucaoFinal = round(monitor[0].width/img.shape[1], 2)
#     img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
#     return img

# model = YOLO("yolov8s.pt")
# cam = cv2.VideoCapture(0)

# count = 0
# fps = 0

# while True:
#     startTimeFPS = time.time()
#     ret, frame = cam.read()
#     img = retifica_imagem(frame)
#     altura = int(img.shape[0]/32)
#     largura = int(img.shape[1]/32)
#     result = model.predict(source=img, imgsz=(altura*32, largura*32), device=0) # imgsz = 320 / 416 / 640 / 960 / 1280 / 1600 / 1920
#     saida = result[0].plot(labels=True)
#     endTimeFPS = time.time()
#     fpsVideo = int(1/(endTimeFPS - startTimeFPS))
#     count += 1
#     if count == 1:
#         fpsMedia = fpsVideo
#     else:
#         fpsMedia = int((fpsMedia * count + fpsVideo) / (count + 1))
#     # Configura e adiciona na imagem textos com os valores de FPS
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(saida, "FPS: " + str(fpsVideo), (10, 30), font, 1, (0, 0, 255), 2)
#     cv2.putText(saida, "Media FPS: " + str(fpsMedia), (10, 60), font, 1, (0, 0, 255), 2)
#     cv2.imshow("Video", saida) # Exibe os frames do vídeo
#     fps += 1
#     if cv2.waitKey(1) == ord('q'):
#         break

# import numpy as np
# import cupy as cp

# x_gpu = cp.array([1, 2, 3])
# print(x_gpu.device)
# print(cp.cuda.runtime.getDeviceCount())

# documentos = ["imagem Espelho"]
# print("\nImagens presente na pasta:")
# i = 1
# for arquivo in documentos:
#     print(f"{i} - {arquivo}")
#     i += 1
# escolha_img = input("Digite o numero da imagem que deseja: ")
# # if escolha_img.isdigit() and int(escolha_img) <= len(documentos) and int(escolha_img) > 0:
# #     print("Tudo certo")
# # else:
# #     print("Algo errado amigo")
# while not(escolha_img.isdigit() and int(escolha_img) <= len(documentos) and int(escolha_img) > 0):
#     print(f"\nSua resposta ({escolha_img}) não corresponde...")
#     print("Imagens presente na pasta:")
#     i = 1
#     for arquivo in documentos:
#         print(f"{i} - {arquivo}")
#         i += 1
#     escolha_img = input("Digite apenas o número da lista de documentos: ")

# print("Fontes disponíveis:\n0 - Câmera\n1 - Vídeo para teste YOLO\n2 - Imagem de teste")
# escolha_cam = input("Qual fonte deseja usar (0, 1, 2)? ")
# while len(escolha_cam) != 1 or not(escolha_cam.isdigit()) or (int(escolha_cam) != 0 and int(escolha_cam) != 1 and int(escolha_cam) != 2):
#     print(f"\nSua resposta ({escolha_cam}) não corresponde: 0, 1, 2\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo para teste YOLO\n2 - Imagem de teste")
#     escolha_cam = input("Qual fonte deseja usar (0, 1, 2)? ")


# import numpy as np
# import cupy as cp
# import cv2
# import time

# def lookup_table(tamanhoFrame):
#     rpixel = Vpn = int(tamanhoFrame)
#     Hpn = int(Vpn*2*np.pi)
#     upn, vpn = np.meshgrid(np.arange(0, Hpn, 1), np.arange(0, Vpn, 1))
#     uimg = (((vpn*rpixel)/Vpn)*np.cos((upn*2*np.pi)/Hpn) + rpixel).astype(np.uint8)
#     vimg = (-(((vpn*rpixel)/Vpn)*np.sin((upn*2*np.pi)/Hpn) - rpixel)).astype(np.uint8)
#     return uimg, vimg

# def retifica_imagem(frame, uimg, vimg):
#     img = np.copy(frame)
#     img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame, centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
#     img = img[vimg, uimg]
#     img = np.flip(img, 0)
#     img = img[corteSup:-corteInf, :]
#     img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
#     return img

# ajusteCentro = (0, -14)
# corteInf = 90
# corteSup = 15
# resolucaoFinal = 1

# imagem_orig = cv2.imread(r"CAMERA\Original\Foto\23-03-05-imagem_Hiperbolica_0.png")

# resolucao = (imagem_orig.shape[1], imagem_orig.shape[0])
# centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
# tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))

# rpixel = Vpn = int(tamanhoFrame)
# Hpn = int(Vpn*2*np.pi)
# uvect = np.arange(0, Hpn, 1)
# vvect = np.arange(0, Vpn, 1)
# upn, vpn = np.meshgrid(uvect, vvect)
# uimg = (((vpn*rpixel)/Vpn)*np.cos((upn*2*np.pi)/Hpn) + rpixel).astype(int)
# vimg = (-(((vpn*rpixel)/Vpn)*np.sin((upn*2*np.pi)/Hpn) - rpixel)).astype(int)
# print(uimg)
# img = np.copy(imagem_orig)
# img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame, centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
# img = img[vimg, uimg]
# img = np.flip(img, 0)
# img = img[corteSup:-corteInf, :]
# img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)

# cv2.imshow("Imagem Original", imagem_orig)
# cv2.imshow("Imagem retificada", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# executar_loop = 1
# while executar_loop:
#     opcao_menu = input("0 - Camera/Video\n1 - Foto/Imagem\n2 - Escolher modelo\n3 - Escolher fonte\n4 - Sair\nDigite sua escolha (0, 1, 2, 3, 4): ")
#     while not(opcao_menu.isdigit()) or (len(opcao_menu) != 1) or (int(opcao_menu) not in [0, 1, 2, 3, 4]):
#         opcao_menu = input(f"\nSua resposta ({opcao_menu}) não corresponde...\n0 - Camera/Video\n1 - Foto/Imagem\n2 - Escolher modelo\n3 - Escolher fonte\n4 - Sair\nDigite sua escolha (0, 1, 2, 3, 4): ")
#     executar_loop = 0

# import os
# from utils import *

# caminho_cam_orig_foto = cria_pasta("CAMERA\Original\Foto")

# print("Transformar imagem")
# documentos = [arquivo for arquivo in os.listdir(caminho_cam_orig_foto) if os.path.isfile(os.path.join(caminho_cam_orig_foto, arquivo))]
# print("\nImagens presente na pasta:")
# i = 1
# for arquivo in documentos:
#     print(f"{i} - {arquivo}")
#     i += 1e
# opcao_escolher_foto = input("Digite o número da imagem que deseja transformar: ")
# while not(opcao_escolher_foto.isdigit()) or (len(opcao_escolher_foto) != 1) or (int(opcao_escolher_foto) not in (range(1, len(documentos)+1))):
#     print(f"\nSua resposta ({opcao_escolher_foto}) não corresponde...")
#     print("Imagens presente na pasta:")
#     i = 1
#     for arquivo in documentos:
#         print(f"{i} - {arquivo}")
#         i += 1
#     opcao_escolher_foto = input("Digite o número da imagem que deseja transformar: ")
# arquivo_imagem = documentos[int(opcao_escolher_foto)-1]
# print(arquivo_imagem)

# from utils import *
# config_camera = cv2.VideoCapture(0)
# ret, frame = config_camera.read()
# start = time.time()
# with stream and device:
#     resolucao = (frame.shape[1], frame.shape[0])
#     centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
#     tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))
#     uimg, vimg = lookup_table(tamanhoFrame)
# end = time.time()
# print(f"{(end-start)*1000:.4f}")
# print(uimg.device)

# lista = []
# for i in range(500):
#     lista.append(i)
# contador = 1
# tamanho_barra = 100
# barra_final = "|"
# for item in lista:
#     progresso = int((100/len(lista))*contador)
#     barra = int((tamanho_barra/100)*progresso)
#     contador += 1
#     os.system("cls")
#     print(f"Vídeo carregando...\n{progresso:3}%|" + "█"*barra + " "*(tamanho_barra-barra) + f"{barra_final}")

# contador = 1
# tamanho_barra = 100
# barra_final = "|"
# for item in lista:
#     progresso = int((100/len(lista))*contador)
#     barra = int((tamanho_barra/100)*progresso)
#     contador += 1
#     os.system("cls")
#     print(f"\nVídeo carregando...\n{progresso:4}% " + "█"*barra)

# dict = {0: 0, 1: 5, 2:10}
# print(len(dict))

# numero = math.ceil(54.32)
# print(numero)
# print(type(numero))

# img = cv2.imread(r"CAMERA\Original\Foto\23-03-05-imagem_Hiperbolica_0.png")
# print(img.shape)

# model = YOLO("yolov8n.pt")
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, largura_camera)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, altura_camera)
# cam.set(cv2.CAP_PROP_FPS, fps_camera)
# cont = 0
# start = time.time()
# while True:
#     ret, frame = cam.read()
#     if ret:
#         results = model.predict(source=frame, imgsz=(736, largura_camera), device=0, stream=True)
#         for img in results:
#             saida = img.plot()
#             cv2.imshow("Video", saida)
#         cont += 1
#         if cv2.waitKey(1) == ord("q"):
#             break
#     else:
#         break
# end = time.time()
# print(f"FPS: {cont/(end-start)}")


# opcao_escolher_foto = input("Digite o número da imagem que deseja transformar: ")
# print(not(opcao_escolher_foto.isdigit()))
# print(len(opcao_escolher_foto) > 2)
# print(len(opcao_escolher_foto) < 1)
# print(int(opcao_escolher_foto) not in (range(1, 18)))
# if not(opcao_escolher_foto.isdigit()) or ((len(opcao_escolher_foto) > 2) and (len(opcao_escolher_foto) < 0)) or (int(opcao_escolher_foto) not in (range(1, 18))):
#     print("\nTa errado")

# camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, altura_camera)
# camera.set(cv2.CAP_PROP_FPS, fps_camera)
# saida_atrasada = None
# modelo_yolo = YOLO("MODELS\yolov8n.pt")

# while True:
#     ret, frame = camera.read()
#     if ret:
#         resultado = modelo_yolo.predict(source=frame, imgsz=640, conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
#         saida = resultado[0].plot(labels=True)

#         if saida_atrasada is not None:
#             print("Calcula distancia")
#             cv2.imshow("Video Atrasado", saida_atrasada)

#         cv2.imshow("Video Atual", saida)
#         saida_atrasada = saida

#         if cv2.waitKey(1) == ord("q"):
#             break
#     else:
#         print("\nCâmera parou de funcionar...")
#         break
# camera.release()

# atrasada = None
# cont = 0
# while True:
#     imagem = np.zeros((900, 900))
#     imagem_num = cv2.putText(imagem, str(cont), (50, 500), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 255, 255), 12)

#     if atrasada is not None:
#         cv2.imshow("Atrasada", atrasada)

#     cv2.imshow("Imagem", imagem_num)

#     atrasada = imagem_num

#     cont += 1
#     if cv2.waitKey(1) == ord("q"):
#         break

# cv2.destroyAllWindows()

# array1 = cp.array([1, 2, 3, 4, 5])
# array2 = array1

# array2 = array2 * 3

# print(f"Array 1: {array1}")
# print(f"Array 2: {array2}")

# print(f"X: {math.sin(math.radians(10)) * 60}")
# print(f"Y: {math.cos(math.radians(10)) * 60}")

# velocidade = input("\nDigite a velocidade que irá manter no carrinho [km/h]: ")
# while not(velocidade.isdigit()) or (len(velocidade) not in [1, 2]) or float(velocidade) <= 0 or float(velocidade) > 20:
#     try:
#         velocidade = float(velocidade)
#         if velocidade <= 0:
#             velocidade = input(f"\nVelocidade digitada ({velocidade}) abaixo ou igual a zero!\n Digite a velocidade que irá manter no carrinho [km/h]: ")
#         elif velocidade > 20:
#             velocidade = input(f"\nVelocidade digitada ({velocidade}) muito alta!\n Digite a velocidade que irá manter no carrinho [km/h]: ")
#     except:
#         velocidade = input(f"\nSua resposta ({velocidade}) não corresponde...\n Digite a velocidade que irá manter no carrinho [km/h]: ")

# velocidade = float(velocidade)
# print(f"\nVelocidade em km/h:    {velocidade:.1f}  km/h")
# velocidade = velocidade / 3.6 # Converte para m/s
# print(f"Velocidade em m/s:     {velocidade:.3f} m/s")
# fps = 15 # Média de 15 FPS nos vídeos
# tempo_entre_foto = 1 / fps
# print(f"1 / {fps} [1/fps]:        {tempo_entre_foto:.3f} s")
# distancia_entre_fotos = velocidade * tempo_entre_foto
# print(f"Distancia entre fotos: {distancia_entre_fotos:.3f} m")

# pos_2_x = float(input(f"\nDigite a posição X da imagem 2 [cm]: ")) / 100
# print(type(pos_2_x))
# print(pos_2_x)
# print(not(pos_2_x.isdigit()))
# print((len(pos_2_x) not in [1, 2, 3]))
# print(int(pos_2_x) < 0)
# while not(pos_2_x.isdigit()) or (len(pos_2_x) not in [1, 2, 3]) or int(pos_2_x) < 0:
#     pos_2_x = input(f"\nSua resposta ({pos_2_x}) não corresponde...\n Digite a posição X da imagem 2 [cm]: ")

# nome_1 = "23-09-18_imagem-localizacao_0.jpg"
# posicao = nome_1.rfind("_") + 1
# ponto = nome_1.rfind(".")
# print(nome_1[posicao : ponto])

# camera = cv2.VideoCapture(0)
# configurar_camera(camera)
# print("\nCâmera configurada")
# gravacao = []
# while True:
#     ret, frame = camera.read()
#     if ret:
#         frame = cv2.putText(frame, "TEXTO", (500, 400),
#                             cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.imshow("Video", frame)
#         gravacao.append(frame)
#         if cv2.waitKey(5) == ord("q"):
#             break
#     else:
#         print("\nQuebrou")
#         break
# cv2.destroyAllWindows()

# altura = gravacao[0].shape[0]
# largura = gravacao[0].shape[1]

# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# video_salvo = cv2.VideoWriter(
#     r"C:\Users\mathe\OneDrive - Instituto Maua de Tecnologia\TCC Lindo\Programa\TCC - testes\video_teste.avi", fourcc, 30, (largura, altura))
# contador = 0
# for imagem in gravacao:
#     frame = imagem.astype("uint8")
#     video_salvo.write(frame)
#     print(f"\nContagem: {contador}")
#     contador += 1

# imagem = cv2.imread(
#     r"YOLO\Foto\23-09-03_imagem-transformada_YOLOv8-nano_03.jpg")
# imagem = cv2.putText(imagem, "Texto aqui porra", (200, 190),
#                      cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
# cv2.imshow("Imagem", imagem)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite(r"C:\Users\mathe\OneDrive - Instituto Maua de Tecnologia\TCC Lindo\Programa\TCC - testes\imagem_teste.jpg",
#             imagem, [cv2.IMWRITE_JPEG_QUALITY, 100])

import cv2
import ultralytics
ultralytics.checks()
model = YOLO("MODELS\yolov8n.pt")
results = model.predict(source=r"C:\Users\mathe\Desktop\bus.jpg", imgsz=640, conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
imagem_saida = results[0].plot(labels=True, font_size=3, line_width=2)

# cv2.imshow("Video YOLO", imagem_saida)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("\n")