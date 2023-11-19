from utils import *


def retifica_imagem(frame, uimg, vimg, centro, tamanhoFrame, resolucaoFinal):
    img = cp.array(frame)
    img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame,
              centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
    img = img[vimg, uimg]
    img = cp.flip(img, 0)
    img = img[corteSup:-corteInf, :]
    img = cp.asnumpy(img, stream=stream)
    img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(
        img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
    return img


def nome_unico_frame_yolo(num_frame, tempo_foto, escolha_modelo):
    caminho_completo_frame = os.path.join(
        caminho_yolo_frame, f"0_{hoje}-frame-{num_frame//tempo_foto}-YOLOv8_{escolha_modelo}.jpg")
    contador = 1
    while os.path.exists(caminho_completo_frame):
        caminho_completo_frame = os.path.join(
            caminho_yolo_frame, f"{contador}_{hoje}-frame-{num_frame//tempo_foto}.jpg")
        contador += 1
    return caminho_completo_frame, caminho_yolo_frame


def nome_unico_frame(num_frame, tempo_foto):
    caminho_completo_frame = os.path.join(
        caminho_cam_transf_frame, f"0_{hoje}-frame-{num_frame//tempo_foto}.jpg")
    contador = 1
    while os.path.exists(caminho_completo_frame):
        caminho_completo_frame = os.path.join(
            caminho_cam_transf_frame, f"{contador}_{hoje}-frame-{num_frame//tempo_foto}.jpg")
        contador += 1
    return caminho_completo_frame, caminho_cam_transf_frame


def yolo_detect(imagem, altura, largura):
    diminuir = 0
    altura_yolo = altura - 32*diminuir
    largura_yolo = largura - 32 * diminuir
    # imgsz = 320 / 416 / 640 / 960 / 1280 / 1600 / 1920
    result = model.predict(source=imagem, imgsz=(
        altura_yolo, largura_yolo), device=0)
    saida = result[0].plot(labels=True)
    return saida


def yolo_modelo():
    modelos = {"nano": "yolov8n.pt", "small": "yolov8s.pt",
               "medium": "yolov8m.pt", "large": "yolov8l.pt", "extra": "yolov8x.pt", }
    print("Tamanhos de modelo da YOLOv8:")
    print("Arquivo\t\tEscolha")
    for chave, valor in modelos.items():
        print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
    escolha_modelo = input("Digite sua escolha: ")
    while escolha_modelo not in modelos.keys():
        print("\nTamanhos de modelo da YOLOv8:")
        print("Arquivo\t\tEscolha")
        for chave, valor in modelos.items():
            print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
        escolha_modelo = input(
            f"\nSua resposta ({escolha_modelo}) não corresponde: {modelos.keys()}\nDigite sua escolha: ")
    print("\nImportando e criando modelo...")
    model = YOLO(modelos[escolha_modelo])
    print("Modelo criado\n")
    time.sleep(1)
    return model, escolha_modelo


def escolha_fonte():
    monitor = get_monitors()
    print("\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo para teste YOLO\n2 - Imagem de teste")
    escolha_cam = input("Qual fonte deseja usar (0, 1, 2)? ")
    while len(escolha_cam) != 1 or not (escolha_cam.isdigit()) or (int(escolha_cam) != 0 and int(escolha_cam) != 1 and int(escolha_cam) != 2):
        print(
            f"\nSua resposta ({escolha_cam}) não corresponde: 0, 1, 2\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo para teste YOLO\n2 - Imagem de teste")
        escolha_cam = input("Qual fonte deseja usar (0, 1, 2)? ")
    escolha_cam = int(escolha_cam)
    print("\nConfigurando câmera e lookup tables...")
    if escolha_cam == 0:
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if not ret:
            print("Não foi possível abrir a câmera.\n\nSaindo...\n")
            exit()
        cam.release()
    elif escolha_cam == 1:
        cam = cv2.VideoCapture(
            r"CAMERA\Original\Video\broadway_-_10836 (540p).mp4")
        ret, frame = cam.read()
        if not ret:
            print("Não foi possível abrir a câmera.\n\nSaindo...\n")
            exit()
        cam.release()
    elif escolha_cam == 2:
        frame = cv2.imread(
            r"CAMERA\Original\Foto\23-03-05-imagem_Hiperbolica_0.png")
    resolucao = (frame.shape[1], frame.shape[0])
    centro = (resolucao[0]//2 + ajusteCentro[0],
              resolucao[1]//2 + ajusteCentro[1])
    tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))
    uimg, vimg = lookup_table(tamanhoFrame)
    with stream and device:
        frame_look = retifica_imagem(
            frame, uimg, vimg, centro, tamanhoFrame, resolucaoFinal=1)
    stream.synchronize()
    resolucaoFinal = round(monitor[0].width/frame_look.shape[1], 2)
    if escolha_cam == 1:
        altura = (int(frame.shape[0]/32))*32
        largura = (int(frame.shape[1]/32))*32
    else:
        altura = (int(frame_look.shape[0]/32))*32
        largura = (int(frame_look.shape[1]/32))*32
    print("Configuração concluída\n")
    time.sleep(1)
    return escolha_cam, altura, largura, centro, tamanhoFrame, resolucaoFinal, uimg, vimg


os.system("cls")

# Cria as pastas caso não existam
print("Criando pastas...")
caminho_cam_orig_foto = cria_pasta("CAMERA\Original\Foto")
caminho_cam_orig_video = cria_pasta("CAMERA\Original\Video")
caminho_cam_transf_foto = cria_pasta("CAMERA\Transformadas\Foto")
caminho_cam_transf_frame = cria_pasta("CAMERA\Transformadas\Frames")
caminho_cam_transf_video = cria_pasta("CAMERA\Transformadas\Video")
caminho_yolo_foto = cria_pasta("YOLO\Foto")
caminho_yolo_frame = cria_pasta("YOLO\Frames")
caminho_yolo_video = cria_pasta("YOLO\Video")
print("Pastas criadas")
time.sleep(1)

escolha_cam, altura, largura, centro, tamanhoFrame, resolucaoFinal, uimg, vimg = escolha_fonte()

model, escolha_modelo = yolo_modelo()

executar = 1
while executar:
    escolha = input(
        "0 - Camera/Video\n1 - Foto/Imagem\n2 - Escolher modelo\n3 - Escolher fonte\n4 - Sair\nDigite sua escolha (0, 1, 2, 3, 4, 5): ")
    if escolha.isdigit() and len(escolha) == 1 and (int(escolha) == 1 or int(escolha) == 0):
        if int(escolha) == 1:
            op_foto = 1
            while op_foto:
                foto = input(
                    "\nO que deseja fazer?\n0 - Tirar uma foto\n1 - Transformar uma imagem existente\nDigite sua escolha (0, 1): ")
                if foto.isdigit() and len(foto) == 1 and (int(foto) == 1 or int(foto) == 0):
                    if int(foto) == 0:
                        foto_salva = 1
                        while foto_salva:  # executar_loop_nova_foto
                            print("\nPreparando a câmera...")
                            cam_foto = cv2.VideoCapture(0)
                            for i in range(5, 0, -1):
                                print(i)
                                time.sleep(1)
                            ret, frame_foto = cam_foto.read()
                            cv2.imshow("Foto", frame_foto)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            cam_foto.release()
                            op_salva = input(
                                "\nO que deseja fazer?\n0 - Salvar a foto\n1 - Tirar outra foto\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                            while not (op_salva.isdigit()) or len(op_salva) != 1 or (int(op_salva) != 2 and int(op_salva) != 1 and int(op_salva) != 0):
                                op_salva = input(
                                    f"\nSua resposta ({op_salva}) nao era esperada: 0, 1, 2\n0 - Salvar foto\n1 - Tirar outra foto\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                            if op_salva == "0":
                                caminho_completo = obter_nome_unico(os.path.join(
                                    caminho_cam_orig_foto, hoje + "-imagem_Hiperbolica.jpg"))
                                cv2.imwrite(caminho_completo, frame_foto, [
                                            cv2.IMWRITE_JPEG_QUALITY, 100])
                                print(
                                    f"\nImagem salva com sucesso em: {caminho_completo}\n\nVoltando...\n")
                                foto_salva = 0
                            elif op_salva == "1":
                                foto_salva = 1
                            elif op_salva == "2":
                                foto_salva = 0
                                print()
                        op_foto = 0
                    elif int(foto) == 1:
                        documentos = [arquivo for arquivo in os.listdir(
                            caminho_cam_orig_foto) if os.path.isfile(os.path.join(caminho_cam_orig_foto, arquivo))]
                        print("\nImagens presente na pasta:")
                        i = 1
                        for arquivo in documentos:
                            print(f"{i} - {arquivo}")
                            i += 1
                        escolha_img = input(
                            "Digite o numero da imagem que deseja: ")
                        while not (escolha_img.isdigit() and int(escolha_img) <= len(documentos) and int(escolha_img) > 0):
                            print(
                                f"\nSua resposta ({escolha_img}) não corresponde...")
                            print("\nImagens presente na pasta:")
                            i = 1
                            for arquivo in documentos:
                                print(f"{i} - {arquivo}")
                                i += 1
                            escolha_img = input(
                                "Digite apenas o número da imagem: ")
                        imagem = documentos[int(escolha_img)-1]
                        start = time.time()
                        # Carrega a imagem
                        frame = cv2.imread(f"{caminho_cam_orig_foto}\{imagem}")
                        with stream and device:
                            img = retifica_imagem(
                                frame, uimg, vimg, centro, tamanhoFrame, resolucaoFinal)
                        stream.synchronize()
                        end = time.time()
                        print(
                            f"\nTempo de execução: {(end-start)*1000:.2f} ms")
                        # Mostra a imagem original
                        cv2.imshow("Imagem Original", frame)
                        cv2.waitKey(0)
                        # Mostra a imagem final
                        cv2.imshow('Imagem Panoramica', img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        op_salva = input(
                            "\nImagem transformada com sucesso. O que deseja fazer agora?\n0 - Salvar imagem transformada\n1 - Passar pela YOLO\n2 - Voltar\nDigite a sua escolha (0, 1, 2): ")
                        while len(op_salva) != 1 or not (op_salva.isdigit()):
                            op_salva = input(
                                f"Sua resposta ({op_salva}) não corresponde...\nO que deseja fazer agora?\n0 - Salvar imagem transformada\n1 - Passar pela YOLO\n2 - Voltar\nDigite a sua escolha (0, 1, 2): ")
                        if int(op_salva) == 0:
                            caminho_completo = obter_nome_unico(os.path.join(
                                caminho_cam_transf_foto, hoje + "-imagem_Transformada.jpg"))
                            cv2.imwrite(caminho_completo, img, [
                                        cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(
                                f"\nImagem salva com sucesso em: {caminho_completo}\n\nVoltando...\n")
                        elif int(op_salva) == 1:
                            print(f"\nPassando imagem para YOLO...")
                            resultado = yolo_detect(img, altura, largura)
                            print()
                            cv2.imshow('Imagem YOLO', resultado)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            caminho_completo = obter_nome_unico(os.path.join(
                                caminho_yolo_foto, hoje + "-imagem_Transformada-YOLOv8_" + escolha_modelo + ".jpg"))
                            cv2.imwrite(caminho_completo, resultado, [
                                        cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(
                                f"Imagem salva com sucesso em: {caminho_completo}\n\nVoltando...\n")
                        elif int(op_salva) == 2:
                            print("\nVoltando...\n")
                        op_foto = 0
                else:
                    print("\nPor favor, digite apenas 0 ou 1...")
                    time.sleep(1)
                    op_foto = 1
            time.sleep(1)
            executar = 1
        elif int(escolha) == 0:
            opGravar = 1
            while opGravar:
                gravar = input("\nDeseja gravar o video (S/n)? ")
                if gravar.isalpha() and len(gravar) == 1 and (gravar.lower() == "s" or gravar.lower() == "n"):
                    if gravar.lower() == "s":
                        print("\nGravar Video -> ATIVADO\n")
                        gravar = 1
                        opGravar = 0
                    elif gravar.lower() == "n":
                        print("\nGravar Video -> DESATIVADO\n")
                        gravar = 0
                        opGravar = 0
                else:
                    print(f"\nSua resposta ({gravar}) não corresponde: S, n")
                    time.sleep(1)
                    opGravar = 1
            optirarFotos = 1
            while optirarFotos:
                tirarFotos = input(
                    "Deseja tirar fotos durante o video (S/n)? ")
                if tirarFotos.isalpha() and len(tirarFotos) == 1 and (tirarFotos.lower() == "s" or tirarFotos.lower() == "n"):
                    if tirarFotos.lower() == "s":
                        print("\nTirar Fotos -> ATIVADO\n")
                        tirarFotos = 1
                        optirarFotos = 0
                    elif tirarFotos.lower() == "n":
                        print("\nTirar Fotos -> DESATIVADO\n")
                        tirarFotos = 0
                        optirarFotos = 0
                else:
                    print(
                        f"\nSua resposta ({tirarFotos}) não corresponde: S, n")
                    time.sleep(1)
                    optirarFotos = 1
            count = 0
            fps = 0
            print("Luz, câmera, ação...")
            if escolha_cam == 0:
                cam = cv2.VideoCapture(0)
            elif escolha_cam == 1:
                cam = cv2.VideoCapture(
                    r"CAMERA\Original\Video\broadway_-_10836 (540p).mp4")
            if not cam.isOpened():
                print("Não foi possível abrir a câmera")
                exit()
            frames_yolo = []
            fotos_yolo = {}
            frames = []
            fotos = {}
            video = []
            tempoInicial = time.time()
            while cam.isOpened():
                startTimeFPS = time.time()  # Inicializa contagem para calcular FPS
                ret, frame = cam.read()  # Realiza a captura do frame do vídeo
                # Transforma a imagem e passa pela YOLO
                if escolha_cam == 0:
                    with stream and device:
                        img = retifica_imagem(
                            frame, uimg, vimg, centro, tamanhoFrame, resolucaoFinal)
                    stream.synchronize()
                    resultado = yolo_detect(img, altura, largura)
                # Apenas detecta pela YOLO no video teste
                elif escolha_cam == 1:
                    img = frame
                    # resultado = yolo_detect(frame, altura, largura)
                endTimeFPS = time.time()
                fpsVideo = int(1/(endTimeFPS - startTimeFPS))
                # Salva uma foto do video a cada quantidade de frames representado por tempo_foto
                tempo_foto = 20
                if (fps % tempo_foto == 0) and tirarFotos:
                    fotos_yolo[fps] = resultado
                    fotos[fps] = img
                if gravar:
                    frames_yolo.append(resultado)
                    frames.append(img)
                    video.append(frame)
                count += 1
                if count == 1:
                    fpsMedia = fpsVideo
                else:
                    fpsMedia = int((fpsMedia * count + fpsVideo) / (count + 1))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(resultado, "FPS: " + str(fpsVideo),
                            (10, 30), font, 1, (0, 0, 255), 2)
                cv2.putText(resultado, "Media FPS: " + str(fpsMedia),
                            (10, 60), font, 1, (0, 0, 255), 2)
                cv2.imshow("Video", resultado)
                fps += 1
                if cv2.waitKey(1) == ord('q'):
                    break
            tempoFinal = time.time()
            print(
                f"\nFrames por segundo: {fps/(tempoFinal - tempoInicial):.2f}")
            cam.release()
            cv2.destroyAllWindows()
            if tirarFotos:
                for fps, frame in fotos_yolo.items():
                    caminho_completo_frame, pasta_frames_yolo = nome_unico_frame_yolo(
                        fps, tempo_foto, escolha_modelo)
                    cv2.imwrite(caminho_completo_frame, frame, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                for fps, frame in fotos.items():
                    caminho_completo_frame, pasta_frames = nome_unico_frame(
                        fps, tempo_foto)
                    cv2.imwrite(caminho_completo_frame, frame, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                print(
                    f"\nImagens transformadas salvas com sucesso na pasta: {pasta_frames}\nImagens com detecção da YOLO salvas com sucesso na pasta: {pasta_frames_yolo}")
            if gravar:
                print(f"\nCriando videos...")
                caminho_completo_gravar_yolo = obter_nome_unico(os.path.join(
                    caminho_yolo_video, hoje + "-video_Transformado-YOLOv8_" + escolha_modelo + ".avi"))
                h, w, _ = frames_yolo[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out = cv2.VideoWriter(
                    caminho_completo_gravar_yolo, fourcc, 30, (w, h))
                for frame in frames_yolo:
                    video_yolo = frame.astype("uint8")
                    out.write(video_yolo)
                out.release()
                caminho_completo_gravar = obter_nome_unico(os.path.join(
                    caminho_cam_transf_video, hoje + "-video_Transformado" + ".avi"))
                h, w, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out = cv2.VideoWriter(
                    caminho_completo_gravar, fourcc, 30, (w, h))
                for frame in frames:
                    video_transf = frame.astype("uint8")
                    out.write(video_transf)
                out.release()
                caminho_completo_gravar_orig = obter_nome_unico(os.path.join(
                    caminho_cam_orig_video, hoje + "-video_Original" + ".avi"))
                h, w, _ = video[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out = cv2.VideoWriter(
                    caminho_completo_gravar_orig, fourcc, 30, (w, h))
                for frame in video:
                    video_orig = frame.astype("uint8")
                    out.write(video_orig)
                out.release()
                print(
                    f"Video original salvo com sucesso em: {caminho_completo_gravar_orig}\nVideo transformado salvo com sucesso em: {caminho_completo_gravar}\nVideo com detecção da YOLO salvo com sucesso em: {caminho_completo_gravar_yolo}")
            print("\nVideo encerrado com sucesso. Saindo...\n")
            time.sleep(1)
            executar = 1
    elif escolha.isdigit() and int(escolha) == 2:
        model, escolha_modelo = yolo_modelo()
        executar = 1
    elif escolha.isdigit() and int(escolha) == 3:
        escolha_cam, altura, largura, centro, tamanhoFrame, resolucaoFinal, uimg, vimg = escolha_fonte()
        executar = 1
    elif escolha.isdigit() and int(escolha) == 4:
        print("\nSaindo...")
        executar = 0
    else:
        print("\nPor favor, digite apenas uma das opções disponíveis...")
        time.sleep(1)
        executar = 1
