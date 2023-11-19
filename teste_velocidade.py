from utils import *


def retifica_imagem_numpy(frame, uimg, vimg):
    img = np.copy(frame)
    img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame,
              centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
    img = img[vimg, uimg]
    img = np.flip(img, 0)
    img = img[corteSup:-corteInf, :]
    resolucaoFinal = round(monitor[0].width/img.shape[1], 2)
    img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(
        img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
    return img


def retifica_imagem_cupy(frame, uimg, vimg, stream):
    img = cp.array(frame)
    img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame,
              centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
    img = img[vimg, uimg]
    img = cp.flip(img, 0)
    img = img[corteSup:-corteInf, :]
    img = cp.asnumpy(img, stream=stream)
    resolucaoFinal = round(monitor[0].width/img.shape[1], 2)
    img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(
        img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
    return img


cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, largura_camera)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, altura_camera)
cam.set(cv2.CAP_PROP_FPS, fps_camera)

if cam.isOpened():
    ret, imagem_orig = cam.read()
resolucao = (imagem_orig.shape[1], imagem_orig.shape[0])
centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))
uimg_cp, vimg_cp = lookup_table_cupy(tamanhoFrame)
uimg_np, vimg_np = lookup_table_numpy(tamanhoFrame)

lista_cupy = []
lista_numpy = []

num = 1000

for i in range(num):

    start_numpy = time.time()
    img_numpy = retifica_imagem_numpy(imagem_orig, uimg_np, vimg_np)
    end_numpy = time.time()
    dif_numpy = (end_numpy - start_numpy)*1000
    lista_numpy.append(dif_numpy)

    start_cupy = time.time()
    with stream and device:
        img_cupy = retifica_imagem_cupy(imagem_orig, uimg_cp, vimg_cp, stream)
    stream.synchronize()
    end_cupy = time.time()
    dif_cupy = (end_cupy - start_cupy)*1000
    lista_cupy.append(dif_cupy)

    os.system("cls")
    print(f"\nFotos transformadas: {i+1}")

cv2.imshow("Imagem Original", imagem_orig)
cv2.imshow("Imagem Numpy", img_numpy)
cv2.imshow("Imagem Cupy", img_cupy)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Imagem\tCupy (ms)\tNumpy (ms)")
for num in range(len(lista_cupy)):
    print(f"{num+1:>6}\t{lista_cupy[num]:>9.4f}\t{lista_numpy[num]:>10.4f}")

print(f"\nComparação de performance para {num} imagens")
print(" "*12 + "\tCupy (ms)\tNumpy (ms)")
print(f"Tempo total\t{sum(lista_cupy):>9.4f}\t{sum(lista_numpy):>10.4f}")
print(f"Melhor tempo\t{min(lista_cupy):>9.4f}\t{min(lista_numpy):>10.4f}")
print(
    f"Tempo medio\t{(sum(lista_cupy)/len(lista_cupy)):>9.4f}\t{(sum(lista_numpy)/len(lista_numpy)):>10.4f}")
print(
    f"FPS\t\t{(num*1000)/(sum(lista_cupy)):>9.4f}\t{(num*1000)/(sum(lista_numpy)):>10.4f}\n")

# fps = 0
# start = time.time()
# while cam.isOpened():
#     ret, frame = cam.read()
#     if ret:
#         with stream and device:
#             img = retifica_imagem_cupy(frame, uimg_cp, vimg_cp, stream)
#         stream.synchronize()
#         cv2.imshow("Video", img)
#         fps += 1
#         if cv2.waitKey(1) == ord('q'):
#             break
#     else:
#         print("Câmera fechou...")
#         break
# end = time.time()
# print(f"Frames por segundo: {fps/(end - start):.2f}")
# cam.release()
