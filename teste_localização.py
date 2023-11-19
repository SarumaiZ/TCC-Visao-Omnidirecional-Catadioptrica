from utils import *

os.system("cls")

largura_imagem = 1922
altura_imagem = 211

num = 50
lista = []

#        (  x,   y,  θ°)
pos_s1 = (0.00, 0.00, 0.00)  # m
pos_s2 = (0.00, 0.50, 0.00)  # m
dx = pos_s2[0] - pos_s1[0]
dy = pos_s2[1] - pos_s1[1]
d_teta = pos_s2[2] - pos_s1[2]
d_teta = d_teta * math.pi / 180

phi_max = 20
phi_min = -40

# Negativo representa esquerda em x e para baixo em y
print(f"\ndx: {dx:.2f} m\tdy: {dy:.2f} m\tdteta: {d_teta:.2f} rad")

modelo_yolo = YOLO(f"MODELS\yolov8n.pt")

imagem_1 = cv2.imread(
    r"C:\Users\mathe\OneDrive - Instituto Maua de Tecnologia\TCC Lindo\Programa\TCC - testes\CAMERA\Transformadas\Foto\23-09-03_imagem-transformada_0.jpg")
imagem_2 = cv2.imread(
    r"C:\Users\mathe\OneDrive - Instituto Maua de Tecnologia\TCC Lindo\Programa\TCC - testes\CAMERA\Transformadas\Foto\23-09-03_imagem-transformada_1.jpg")

img1_saida = modelo_yolo.predict(source=imagem_1, imgsz=(
    192, 1760), conf=0.30, device=0, classes=classes_yolo, max_det=50)
img1_loc = img1_saida[0].plot(labels=False)
boxes1 = img1_saida[0].boxes.cpu().numpy()

img2_saida = modelo_yolo.predict(source=imagem_2, imgsz=(
    192, 1760), conf=0.30, device=0, classes=classes_yolo, max_det=50)
img2_loc = img2_saida[0].plot(labels=False)
boxes2 = img2_saida[0].boxes.cpu().numpy()

for i in range(num):
    start = time.time()
    with stream and device:
        s2_width = cp.array(boxes2.xywh.astype(int)[:, 2])
        s2_height = cp.array(boxes2.xywh.astype(int)[:, 3])

        s2_altura = cp.array(boxes2.xywh.astype(int)[:, 1])

        s1_teta = cp.array(boxes1.xywh.astype(int)[:, 0])
        s2_teta = cp.array(boxes2.xywh.astype(int)[:, 0])
        s1_teta = s1_teta * 2 * math.pi / largura_imagem

        # Pra mim, faz mais sentido subtrair do q somar
        s2_teta = (s2_teta * 2 * math.pi / largura_imagem) + d_teta
        m1 = s1_teta.shape[0]
        m2 = s2_teta.shape[0]
        m_max = max(m1, m2)
        m_min = min(m1, m2)
        zeros = cp.zeros((m_max))
        if m1 > m2:
            zeros[:s2_teta.shape[0]] = s2_teta
            s2_teta = zeros
        elif m2 > m1:
            zeros[:s1_teta.shape[0]] = s1_teta
            s1_teta = zeros
        matriz1 = cp.repeat(s1_teta.reshape(1, -1), m_max, axis=0)
        matriz2 = cp.repeat(s2_teta.reshape(1, -1), m_max, axis=0)
        delta_teta = cp.abs(matriz1 - matriz2.T)
        # Matriz delta: colunas se refere ao vetor 1 e linhas ao vetor 2
        indices = cp.argmin(delta_teta, axis=1)
        indices = indices[:m_min]
        s1_teta = s1_teta[:m_min]
        s2_teta = s2_teta[:m_min]

        s2_altura = s2_altura[:m_min]
        s2_height = s2_height[:m_min]
        s2_width = s2_width[:m_min]

        s1_teta = s1_teta[indices]

        X_p = (dx * cp.tan(s2_teta) - dy) / (cp.tan(s2_teta) - cp.tan(s1_teta))
        Y_p = X_p * cp.tan(s1_teta)

        distancia_objs = cp.sqrt(cp.power(X_p, 2) + cp.power(Y_p, 2))

        # criar dois vetores ao inves de voltar
        s2_teta_pixel = (((s2_teta - d_teta) * largura_imagem) /
                         (2 * math.pi)).astype(int)
        distancia_objs = cp.round(distancia_objs, 2)

        s2_z = altura_imagem - s2_altura
        phi = ((s2_z / altura_imagem) * (phi_max - phi_min)) + phi_min
        Z_p = distancia_objs * cp.tan(phi)

    stream.synchronize()
    end = time.time()
    lista.append((end-start)*1000)

print(f"\nMatriz 1: \n{matriz1}\n\nMatriz 2: \n{matriz2}")
print(f"\nMatriz delta: \n{delta_teta}")
print(f"\nMatriz Indices: \n{indices}")
print(f"\nVetor 1 corrigido: \n{s1_teta}\nVetor 2 corrigido: \n{s2_teta}")
print(
    f"\nVetor posição X: \n{X_p}\nVetor posição Y: \n{Y_p}\nVetor posição Z: \n{Z_p}")
print(f"\nVetor distancia: \n{distancia_objs}")
print(f"\n\nTempo total:  {sum(lista[1:])} ms")
print(f"Melhor tempo: {min(lista[1:])} ms")
print(f"Tempo medio: {(sum(lista[1:]))/(len(lista[1:]))} ms")
print(f"Primeira execução: {lista[1]}\nÚltima execução: {lista[-1]}")

# texto_distancia = list(map(str, distancia_objs))
# print()
# print(texto_distancia)

for i in range(len(distancia_objs)):
    img2_loc = cv2.putText(img2_loc,
                           str(distancia_objs[i]) + "m",
                           (int(s2_teta_pixel[i] - (s2_width[i]/2) + 5),
                            int(s2_altura[i] + (s2_height[i]/2) - 5)),
                           cv2.FONT_HERSHEY_DUPLEX,
                           0.6,
                           (0, 0, 255),
                           2,
                           cv2.LINE_AA)

cv2.imshow("Imagem 1", img1_loc)
cv2.waitKey(0)
cv2.imshow("Imagem 2", img2_loc)
cv2.waitKey(0)
cv2.destroyAllWindows()
