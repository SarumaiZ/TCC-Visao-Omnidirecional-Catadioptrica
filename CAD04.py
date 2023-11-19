from utils import *
print("\nImportando funções e constantes...")

print("\nCriando pastas...")
caminho_cam_orig_foto = cria_pasta("CAMERA\Original\Foto")
caminho_cam_orig_video = cria_pasta("CAMERA\Original\Video")
caminho_cam_transf_foto = cria_pasta("CAMERA\Transformadas\Foto")
caminho_cam_transf_frame = cria_pasta("CAMERA\Transformadas\Frames")
caminho_cam_transf_video = cria_pasta("CAMERA\Transformadas\Video")
caminho_cam_transf_dist = cria_pasta("CAMERA\Transformadas\Localizacao")
caminho_modelos = cria_pasta("MODELS")
caminho_yolo_foto = cria_pasta("YOLO\Foto")
caminho_yolo_frame = cria_pasta("YOLO\Frames")
caminho_yolo_video = cria_pasta("YOLO\Video")


def retifica_imagem(frame, uimg, vimg, stream):
    img = cp.array(frame)
    img = img[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame,
              centro[0]-tamanhoFrame:centro[0]+tamanhoFrame]
    img = img[vimg, uimg]
    img = cp.flip(img, 0)
    img = img[corteSup:-corteInf, :]
    img = cp.asnumpy(img, stream=stream)
    resolucaoFinal = round(monitor[0].width/img.shape[1], 2)
    img = cv2.resize(img, (int(img.shape[1]*resolucaoFinal), int(img.shape[0]*resolucaoFinal)), interpolation=cv2.INTER_CUBIC)
    return img


cv_cam = 1

print("\nConfigurando câmera e variáveis...")
config_camera = cv2.VideoCapture(cv_cam)
configurar_camera(config_camera)
ret, frame = config_camera.read()
if not ret:
    print("\nNão foi possível abrir a câmera USB...\nUtilizando webcam...")
    cv_cam = 0
    config_camera.release()
    config_camera = cv2.VideoCapture(cv_cam)
    configurar_camera(config_camera)
    ret, frame = config_camera.read()
resolucao = (frame.shape[1], frame.shape[0])
centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))
uimg, vimg = lookup_table(tamanhoFrame)
with stream and device:
    config_frame = retifica_imagem(frame, uimg, vimg, stream)
stream.synchronize()
altura_transf = config_frame.shape[0]
largura_transf = config_frame.shape[1]
print(f"Largura: {largura_transf}\nAltura: {altura_transf}")
altura_yolo = (int(config_frame.shape[0]/32))*32
largura_yolo = (int(config_frame.shape[1]/32))*32
config_camera.release()
print("Configuração concluída!")
time.sleep(1)

modelos_dict = {"nano": "yolov8n.pt", "small": "yolov8s.pt",
                "medium": "yolov8m.pt", "large": "yolov8l.pt", "extra": "yolov8x.pt", }
print("\nTamanhos de modelo da YOLOv8:")
print("Arquivo\t\tEscolha")
for chave, valor in modelos_dict.items():
    print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
escolha_modelo_yolo = input("Digite sua escolha: ")
while escolha_modelo_yolo not in modelos_dict.keys():
    print("\nTamanhos de modelo da YOLOv8:")
    print("Arquivo\t\tEscolha")
    for chave, valor in modelos_dict.items():
        print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
    escolha_modelo_yolo = input(
        f"\nSua resposta ({escolha_modelo_yolo}) não corresponde: {modelos_dict.keys()}\nDigite sua escolha: ")
print("\nImportando e criando modelo...")
modelo_yolo = YOLO(f"MODELS\{modelos_dict[escolha_modelo_yolo]}")
print("Importação concuída!")
time.sleep(1)

os.system("cls")

executar_loop = 1
while executar_loop:
    opcao_menu = input(
        "\nOpções disponíveis:\n0 - Camera/Video\n1 - Foto/Imagem\n2 - Escolher modelo\n3 - Calcular localização\n4 - Sair\nDigite sua escolha (0, 1, 2, 3, 4): ")
    while not (opcao_menu.isdigit()) or len(opcao_menu) != 1 or (int(opcao_menu) not in [0, 1, 2, 3, 4]):
        opcao_menu = input(
            f"\nSua resposta ({opcao_menu}) não corresponde...\n0 - Camera/Video\n1 - Foto/Imagem\n2 - Escolher modelo\n3 - Calcular localização\n4 - Sair\nDigite sua escolha (0, 1, 2, 3, 4): ")
    opcao_menu = int(opcao_menu)
    match opcao_menu:

        case 0:
            opcao_gravar = input(
                "\nGravar vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
            while not (opcao_gravar.isdigit()) or len(opcao_gravar) != 1 or (int(opcao_gravar) not in [0, 1]):
                opcao_gravar = input(
                    f"\nSua resposta ({opcao_gravar}) não corresponde...\nGravar vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
            opcao_gravar = int(opcao_gravar)
            match opcao_gravar:
                case 0:
                    print("\nGravar vídeo -> DESATIVADO")
                case 1:
                    print("\nGravar vídeo -> ATIVADO")

            opcao_captura_frame = input(
                "\nCapturar frames ao longo do vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
            while not (opcao_captura_frame.isdigit()) or len(opcao_captura_frame) != 1 or (int(opcao_captura_frame) not in [0, 1]):
                opcao_captura_frame = input(
                    f"\nSua resposta ({opcao_captura_frame}) não corresponde...\nCapturar frames ao longo do vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
            opcao_captura_frame = int(opcao_captura_frame)
            match opcao_captura_frame:
                case 0:
                    print("\nCapturar frames -> DESATIVADO")
                case 1:
                    print("\nCapturar frames -> ATIVADO")

            opcao_camera_video = input(
                "\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo armazenado\nDigite sua escolha (0, 1): ")
            while not (opcao_camera_video.isdigit()) or len(opcao_camera_video) != 1 or (int(opcao_camera_video) not in [0, 1]):
                opcao_camera_video = input(
                    f"\nSua resposta ({opcao_camera_video}) não corresponde...\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo armazenado\nDigite sua escolha (0, 1): ")
            opcao_camera_video = int(opcao_camera_video)
            match opcao_camera_video:
                case 0:
                    try:
                        camera = cv2.VideoCapture(cv_cam)
                        configurar_camera(camera)
                    except:
                        print("\nNão foi possível abrir a câmera")
                        break
                case 1:
                    videos = [arquivo for arquivo in os.listdir(
                        caminho_cam_orig_video) if os.path.isfile(os.path.join(caminho_cam_orig_video, arquivo))]
                    print("\nVídeos presentes na pasta:")
                    i = 1
                    for video in videos:
                        print(f"{i:>2} - {video}")
                        i += 1
                    opcao_escolher_video = input(
                        "Digite o número do video que deseja: ")
                    while not (opcao_escolher_video.isdigit()) or ((len(opcao_escolher_video) > 2) and (len(opcao_escolher_video) < 0)) or (int(opcao_escolher_video) not in (range(1, len(videos)+1))):
                        print(
                            f"\nSua reposta ({opcao_escolher_video}) não corresponde...")
                        print("\nVídeos presentes na pasta:")
                        i = 1
                        for video in videos:
                            print(f"{i:>2} - {video}")
                            i += 1
                        opcao_escolher_video = input(
                            "Digite o número do video que deseja: ")
                    arquivo_video = videos[int(opcao_escolher_video) - 1]
                    camera = cv2.VideoCapture(
                        f"{caminho_cam_orig_video}\{arquivo_video}")
            fps = 0
            captura_frame_yolo = {}
            captura_frame = {}
            video_original = []
            video_transformado = []
            video_transf_yolo = []
            print("\n")
            start_time = time.time()
            while True:
                ret, frame = camera.read()
                if ret:
                    with stream and device:
                        #frame = cp.flip(frame, 1)
                        imagem = retifica_imagem(frame, uimg, vimg, stream)
                        imagem = cp.flip(imagem, 1)
                        imagem_resultado = modelo_yolo.predict(source=imagem, imgsz=(
                            altura_yolo, largura_yolo), conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
                        imagem_saida = imagem_resultado[0].plot(
                            labels=True, font_size=3, line_width=2)
                        cv2.imshow("Video Original", frame)
                        cv2.imshow("Video YOLO", imagem_saida)
                        if opcao_captura_frame and fps % tempo_captura_frame == 0:
                            captura_frame_yolo[fps] = imagem_saida
                            captura_frame[fps] = imagem
                        if opcao_gravar:
                            video_original.append(frame)
                            video_transformado.append(imagem)
                            video_transf_yolo.append(imagem_saida)
                    stream.synchronize()
                    fps += 1
                    if cv2.waitKey(1) == ord("q"):
                        break
                else:
                    print("\nCâmera parou de funcionar...")
                    break
            end_time = time.time()
            camera.release()
            cv2.destroyAllWindows()
            fps_video = fps/(end_time - start_time)
            print(f"\nFrames por segundo: {fps_video:.2f}")
            fps_video = math.ceil(fps_video)
            print("\nVídeo encerrado com sucesso!")

            if opcao_gravar:
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                print("\nCriando vídeos...\n\nAVISO: Isso pode demorar alguns minutos!")
                time.sleep(3)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                caminho_completo_gravar_orig = os.path.join(
                    caminho_cam_orig_video, f"{hoje}_video-original.avi")
                caminho_completo_gravar_orig = obter_nome_unico(
                    caminho_completo_gravar_orig)
                altura_video_original = video_original[0].shape[0]
                largura_video_original = video_original[0].shape[1]
                video_saida = cv2.VideoWriter(
                    caminho_completo_gravar_orig, fourcc, fps_video, (largura_video_original, altura_video_original))
                contador = 1
                for frame in video_original:
                    frame_video_original = frame.astype("uint8")
                    video_saida.write(frame_video_original)
                    progresso_video = int((100/len(video_original))*contador)
                    contador += 1
                    os.system("cls")
                    print(f"Vídeo original carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                        100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_original)}]")
                video_saida.release()
                print("Vídeo original criado!")
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                caminho_completo_gravar_transf = os.path.join(
                    caminho_cam_transf_video, f"{hoje}_video-transformado.avi")
                caminho_completo_gravar_transf = obter_nome_unico(
                    caminho_completo_gravar_transf)
                altura_video_transf = video_transformado[0].shape[0]
                largura_video_transf = video_transformado[0].shape[1]
                video_saida = cv2.VideoWriter(
                    caminho_completo_gravar_transf, fourcc, fps_video, (largura_video_transf, altura_video_transf))
                contador = 1
                for frame in video_transformado:
                    frame_video_transf = frame.astype("uint8")
                    video_saida.write(frame_video_transf)
                    progresso_video = int(
                        (100/len(video_transformado))*contador)
                    contador += 1
                    os.system("cls")
                    print(f"Vídeo retificado carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                        100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_transformado)}]")
                video_saida.release()
                print("Vídeo retificado criado!")
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                caminho_completo_gravar_yolo = os.path.join(
                    caminho_yolo_video, f"{hoje}_video-transformado_YOLOv8-{escolha_modelo_yolo}.avi")
                caminho_completo_gravar_yolo = obter_nome_unico(
                    caminho_completo_gravar_yolo)
                altura_video_yolo = video_transf_yolo[0].shape[0]
                largura_video_yolo = video_transf_yolo[0].shape[1]
                video_saida = cv2.VideoWriter(
                    caminho_completo_gravar_yolo, fourcc, fps_video, (largura_video_transf, altura_video_transf))
                contador = 1
                for frame in video_transf_yolo:
                    frame_video_yolo = frame.astype("uint8")
                    video_saida.write(frame_video_yolo)
                    progresso_video = int(
                        (100/len(video_transf_yolo))*contador)
                    contador += 1
                    os.system("cls")
                    print(f"Vídeo com detecção da YOLO carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                        100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_transf_yolo)}]")
                video_saida.release()
                print("Vídeo com detecção da YOLO criado!")
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                os.system("cls")
                print(
                    f"Vídeo original salvo com sucesso:\n {caminho_completo_gravar_orig}")
                print(
                    f"\nVídeo retificado salvo com sucesso:\n {caminho_completo_gravar_transf}")
                print(
                    f"\nVídeo com detecção da YOLO salvo com sucesso:\n {caminho_completo_gravar_yolo}")

            if opcao_captura_frame:
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                print(
                    "\nArmazenando frames do vídeo...\n\nAVISO: Isso pode demorar alguns segundos!")
                time.sleep(3)
                contador = 1
                for fps, frame_transf in captura_frame.items():
                    if (fps//tempo_captura_frame) < 10:
                        numero_frame = "0" + str(fps//tempo_captura_frame)
                    else:
                        numero_frame = str(fps//tempo_captura_frame)
                    caminho_completo_frame_transf = os.path.join(
                        caminho_cam_transf_frame, f"{hoje}_video-00_frame-{numero_frame}.jpg")
                    contador_caminho = 1
                    while os.path.exists(caminho_completo_frame_transf):
                        if contador_caminho < 10:
                            contador_caminho = "0" + str(contador_caminho)
                        caminho_completo_frame_transf = os.path.join(
                            caminho_cam_transf_frame, f"{hoje}_video-{contador_caminho}_frame-{numero_frame}.jpg")
                        contador_caminho += 1

                    cv2.imwrite(caminho_completo_frame_transf, frame_transf, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                    progresso_captura = int((100/len(captura_frame))*contador)
                    contador += 1
                    os.system("cls")
                    print(f"Armazenando frames capturados do vídeo transformado...\n{progresso_captura:4}% |" + "█"*progresso_captura + " "*(
                        100-progresso_captura) + f"| Frames carregados: [{contador-1}/{len(captura_frame)}]")
                print("Frames do vídeo retificado armazenados!")
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                contador = 1
                for fps, frame_yolo in captura_frame_yolo.items():
                    if (fps//tempo_captura_frame) < 10:
                        numero_frame = "0" + str(fps//tempo_captura_frame)
                    else:
                        numero_frame = str(fps//tempo_captura_frame)
                    caminho_completo_frame_yolo = os.path.join(
                        caminho_yolo_frame, f"{hoje}_video-00_frame-{numero_frame}_YOLOv8-{escolha_modelo_yolo}.jpg")
                    contador_caminho = 1
                    while os.path.exists(caminho_completo_frame_yolo):
                        if contador_caminho < 10:
                            contador_caminho = "0" + str(contador_caminho)
                        caminho_completo_frame_yolo = os.path.join(
                            caminho_yolo_frame, f"{hoje}_video-{contador_caminho}_frame-{numero_frame}_YOLOv8-{escolha_modelo_yolo}.jpg")
                        contador_caminho += 1

                    cv2.imwrite(caminho_completo_frame_yolo, frame_yolo, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                    progresso_captura = int(
                        (100/len(captura_frame_yolo))*contador)
                    contador += 1
                    os.system("cls")
                    print(f"Armazenando frames capturados do vídeo com detecção da YOLO...\n{progresso_captura:4}% |" + "█"*progresso_captura + " "*(
                        100-progresso_captura) + f"| Frames carregados: [{contador-1}/{len(captura_frame_yolo)}]")
                print("Frames do vídeo com detecção da YOLO armazenados!")
                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                os.system("cls")

                print(
                    f"Frames retificados do vídeo armazenados na pasta:\n {caminho_cam_transf_frame}")
                print(
                    f"\nFrames retificados com detecção da YOLO armazenados na pasta:\n {caminho_yolo_frame}")

                opcao_continua = input(
                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                os.system("cls")
            print("\nVoltando...")
            executar_loop = 1

        case 1:
            opcao_foto = input(
                "\nO que deseja fazer:\n0 - Tirar uma nova foto\n1 - Transformar uma imagem existente\nDigite sua escolha (0, 1): ")
            while not (opcao_foto.isdigit()) or len(opcao_foto) != 1 or (int(opcao_foto) not in [0, 1]):
                opcao_foto = input(
                    f"\nSua resposta ({opcao_foto}) não corresponde...\nO que deseja fazer:\n0 - Tirar uma nova foto\n1 - Transformar uma imagem existente\nDigite sua escolha (0, 1): ")
            opcao_foto = int(opcao_foto)
            match opcao_foto:

                case 0:
                    executar_loop_nova_foto = 1
                    while executar_loop_nova_foto:
                        print("\nPreparando câmera...")
                        camera_foto = cv2.VideoCapture(cv_cam)
                        configurar_camera(camera_foto)
                        i = 0
                        while i < 10:
                            ret, frame_foto = camera_foto.read()
                            i += 1
                            # print(f"Contagem: {i}")
                        print("Câmera configurada!\n\nContagem regressiva...")
                        for i in range(5, 0, -1):
                            print(i)
                            time.sleep(1)
                        ret, frame_foto = camera_foto.read()
                        frame_foto = np.flip(frame_foto, 1)
                        cv2.imshow("Foto", frame_foto)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        camera_foto.release()
                        opcao_salva_foto = input(
                            "\nO que deseja fazer:\n0 - Salvar a foto\n1 - Tirar uma nova foto\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                        while not (opcao_salva_foto.isdigit()) or len(opcao_salva_foto) != 1 or (int(opcao_salva_foto) not in [0, 1, 2]):
                            opcao_salva_foto = input(
                                f"\nSua resposta ({opcao_salva_foto}) não corresponde...\nO que deseja fazer:\n0 - Salvar a foto\n1 - Tirar uma nova foto\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                        opcao_salva_foto = int(opcao_salva_foto)
                        match opcao_salva_foto:

                            case 0:
                                caminho_completo_foto = os.path.join(
                                    caminho_cam_orig_foto, f"{hoje}_imagem-hiperbolica.jpg")
                                caminho_completo_foto = obter_nome_unico(
                                    caminho_completo_foto)
                                cv2.imwrite(caminho_completo_foto, frame_foto, [
                                            cv2.IMWRITE_JPEG_QUALITY, 100])
                                print(
                                    f"\nFoto salva com sucesso:\n {caminho_completo_foto}")
                                opcao_continua = input(
                                    "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                                os.system("cls")
                                executar_loop_nova_foto = 0

                            case 1:
                                executar_loop_nova_foto = 1

                            case 2:
                                print("\nVoltando...")
                                time.sleep(2)
                                os.system("cls")
                                executar_loop_nova_foto = 0

                case 1:
                    documentos = [arquivo for arquivo in os.listdir(
                        caminho_cam_orig_foto) if os.path.isfile(os.path.join(caminho_cam_orig_foto, arquivo))]
                    print("\nImagens presente na pasta:")
                    i = 1
                    for arquivo in documentos:
                        print(f"{i:>2} - {arquivo}")
                        i += 1
                    opcao_escolher_foto = input(
                        "Digite o número da imagem que deseja transformar: ")
                    while not (opcao_escolher_foto.isdigit()) or ((len(opcao_escolher_foto) > 2) and (len(opcao_escolher_foto) < 0)) or (int(opcao_escolher_foto) not in (range(1, len(documentos)+1))):
                        print(
                            f"\nSua resposta ({opcao_escolher_foto}) não corresponde...")
                        print("Imagens presente na pasta:")
                        i = 1
                        for arquivo in documentos:
                            print(f"{i:>2} - {arquivo}")
                            i += 1
                        opcao_escolher_foto = input(
                            "Digite o número da imagem que deseja transformar: ")
                    arquivo_imagem = documentos[int(opcao_escolher_foto)-1]
                    start_time = time.time()
                    frame_foto_transf = cv2.imread(
                        f"{caminho_cam_orig_foto}\{arquivo_imagem}")
                    with stream and device:
                        imagem_transf = retifica_imagem(
                            frame_foto_transf, uimg, vimg, stream)
                    stream.synchronize()
                    end_time = time.time()
                    print(
                        f"\nTempo para retificação: {(end_time - start_time)*1000} ms")
                    cv2.imshow("Imagem Original", frame_foto_transf)
                    cv2.waitKey(0)
                    cv2.imshow("Imagem Retificada", imagem_transf)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    opcao_salva_foto_transformada = input(
                        "\nImagem transformada com sucesso. O que deseja fazer:\n0 - Salvar a imagem transformada\n1 - Utilizar a YOLO para detecção\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                    while not (opcao_salva_foto_transformada.isdigit()) or len(opcao_salva_foto_transformada) != 1 or (int(opcao_salva_foto_transformada) not in [0, 1, 2]):
                        opcao_salva_foto_transformada = input(
                            f"\nSua resposta ({opcao_salva_foto_transformada}) não corresponde...\nImagem transformada com sucesso. O que deseja fazer:\n0 - Salvar a imagem transformada\n1 - Utilizar a YOLO para detecção\n2 - Voltar\nDigite sua escolha (0, 1, 2): ")
                    opcao_salva_foto_transformada = int(
                        opcao_salva_foto_transformada)
                    match opcao_salva_foto_transformada:

                        case 0:
                            caminho_completo_foto_transf = os.path.join(
                                caminho_cam_transf_foto, f"{hoje}_imagem-transformada.jpg")
                            caminho_completo_foto_transf = obter_nome_unico(
                                caminho_completo_foto_transf)
                            cv2.imwrite(caminho_completo_foto_transf, imagem_transf, [
                                        cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(
                                f"\nImagem transformada salva com sucesso:\n {caminho_completo_foto_transf}")

                            print("\nVoltando...")
                            time.sleep(0.3)

                        case 1:
                            print("\nInicializando YOLO e enviando imagem...")
                            imagem_resultado = modelo_yolo.predict(source=imagem_transf, imgsz=(
                                altura_yolo, largura_yolo), conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
                            imagem_saida = imagem_resultado[0].plot(
                                labels=False, font_size=3, line_width=2)
                            boxes = imagem_resultado[0].boxes.cpu().numpy()
                            for i in range(len(boxes)):
                                box = boxes[i].xywh[0].astype(int)
                                imagem_saida = cv2.circle(
                                    imagem_saida, (box[0], box[1]), 2, (0, 255, 0), 5)
                            print(boxes.xywh.astype(int))
                            print(boxes.xywh.astype(int)[:, 0])
                            cv2.imshow("YOLO", imagem_saida)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            print("\nSalvando imagem com detecção...")
                            caminho_completo_transf_yolo = os.path.join(
                                caminho_yolo_foto, f"{hoje}_imagem-transformada_YOLOv8-{escolha_modelo_yolo}.jpg")
                            caminho_completo_transf_yolo = obter_nome_unico(
                                caminho_completo_transf_yolo)
                            cv2.imwrite(caminho_completo_transf_yolo, imagem_saida, [
                                        cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(
                                f"\nImagem com detecção YOLO salva com sucesso:\n {caminho_completo_transf_yolo}")

                            print("\nVoltando...")
                            time.sleep(0.3)

                        case 2:
                            print("\nVoltando...")
                            time.sleep(0.3)

            executar_loop = 1

        case 2:
            print("\nTamanhos de modelo da YOLOv8:")
            print("Arquivo\t\tEscolha")
            for chave, valor in modelos_dict.items():
                print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
            escolha_modelo_yolo = input("Digite sua escolha: ")
            while escolha_modelo_yolo not in modelos_dict.keys():
                print("\nTamanhos de modelo da YOLOv8:")
                print("Arquivo\t\tEscolha")
                for chave, valor in modelos_dict.items():
                    print(f"{valor[:-6].upper() + valor[-6:]}\t{chave}")
                escolha_modelo_yolo = input(
                    f"\nSua resposta ({escolha_modelo_yolo}) não corresponde: {modelos_dict.keys()}\nDigite sua escolha: ")
            print("\nImportando e criando modelo...")
            modelo_yolo = YOLO(f"MODELS\{modelos_dict[escolha_modelo_yolo]}")
            print("Importação concuída!")
            time.sleep(1)

            executar_loop = 1

        case 3:
            opcao_local = input(
                "\nO que deseja fazer:\n0 - Distância dos objetos em vídeo\n1 - Utilizar duas imagens existentes\nDigite sua escolha (0, 1): ")
            while not (opcao_local.isdigit()) or len(opcao_local) != 1 or (int(opcao_local) not in [0, 1]):
                opcao_local = input(
                    f"\nSua resposta ({opcao_local}) não corresponde...\nO que deseja fazer:\n0 - Distância dos objetos em vídeo\n1 - Utilizar duas imagens existentes\nDigite sua escolha (0, 1): ")
            opcao_local = int(opcao_local)

            match opcao_local:

                case 0:
                    print("\nUitlizando localização de objetos em vídeo...")
                    time.sleep(0.3)
                    opcao_gravar_loc = input(
                        "\nGravar vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
                    while not (opcao_gravar_loc.isdigit()) or len(opcao_gravar_loc) != 1 or (int(opcao_gravar_loc) not in [0, 1]):
                        opcao_gravar_loc = input(
                            f"\nSua resposta ({opcao_gravar_loc}) não corresponde...\nGravar vídeo:\n0 - DESATIVAR\n1 - ATIVAR\nDigite sua escolha (0, 1): ")
                    opcao_gravar_loc = int(opcao_gravar_loc)
                    match opcao_gravar_loc:
                        case 0:
                            print("\nGravar vídeo -> DESATIVADO")
                        case 1:
                            print("\nGravar vídeo -> ATIVADO")

                    time.sleep(0.3)
                    velocidade = input(
                        "\nDigite a velocidade que irá manter no carrinho [km/h]: ")
                    while not (velocidade.isdigit()) or (len(velocidade) not in [1, 2]) or float(velocidade) <= 0 or float(velocidade) > 20:
                        try:
                            velocidade = float(velocidade)
                            if velocidade <= 0:
                                velocidade = input(
                                    f"\nVelocidade digitada ({velocidade}) abaixo ou igual a zero!\n Digite a velocidade que irá manter no carrinho [km/h]: ")
                            elif velocidade > 20:
                                velocidade = input(
                                    f"\nVelocidade digitada ({velocidade}) muito alta!\n Digite a velocidade que irá manter no carrinho [km/h]: ")
                        except:
                            velocidade = input(
                                f"\nSua resposta ({velocidade}) não corresponde...\n Digite a velocidade que irá manter no carrinho [km/h]: ")

                    velocidade = float(velocidade)
                    print(f"\nVelocidade em km/h:    {velocidade:.1f}  km/h")
                    velocidade = velocidade / 3.6  # Converte para m/s
                    print(f"Velocidade em m/s:     {velocidade:.3f} m/s")
                    fps = 15  # Média de 10 FPS nos vídeos com localização
                    tempo_entre_foto = 1 / fps
                    print(f"1 / {fps} [1/fps]:        {tempo_entre_foto:.3f} s")
                    distancia_entre_fotos = velocidade * tempo_entre_foto
                    print(f"Distancia entre fotos: {distancia_entre_fotos:.3f} m")
                    time.sleep(1)

                    opcao_camera_video = input(
                        "\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo armazenado\nDigite sua escolha (0, 1): ")
                    while not (opcao_camera_video.isdigit()) or len(opcao_camera_video) != 1 or (int(opcao_camera_video) not in [0, 1]):
                        opcao_camera_video = input(
                            f"\nSua resposta ({opcao_camera_video}) não corresponde...\nFontes disponíveis:\n0 - Câmera\n1 - Vídeo armazenado\nDigite sua escolha (0, 1): ")
                    opcao_camera_video = int(opcao_camera_video)
                    match opcao_camera_video:
                        case 0:
                            try:
                                camera = cv2.VideoCapture(cv_cam)
                                configurar_camera(camera)
                            except:
                                print("\nNão foi possível abrir a câmera")
                                break
                        case 1:
                            videos = [arquivo for arquivo in os.listdir(
                                caminho_cam_orig_video) if os.path.isfile(os.path.join(caminho_cam_orig_video, arquivo))]
                            print("\nVídeos presentes na pasta:")
                            i = 1
                            for video in videos:
                                print(f"{i:>2} - {video}")
                                i += 1
                            opcao_escolher_video = input(
                                "Digite o número do video que deseja: ")
                            while not (opcao_escolher_video.isdigit()) or ((len(opcao_escolher_video) > 2) and (len(opcao_escolher_video) < 0)) or (int(opcao_escolher_video) not in (range(1, len(videos)+1))):
                                print(
                                    f"\nSua reposta ({opcao_escolher_video}) não corresponde...")
                                print("\nVídeos presentes na pasta:")
                                i = 1
                                for video in videos:
                                    print(f"{i:>2} - {video}")
                                    i += 1
                                opcao_escolher_video = input(
                                    "Digite o número do video que deseja: ")
                            arquivo_video = videos[int(
                                opcao_escolher_video) - 1]
                            camera = cv2.VideoCapture(
                                f"{caminho_cam_orig_video}\{arquivo_video}")

                    fps = 0
                    video_original_loc = []
                    video_transformado_loc = []
                    video_transf_yolo_loc = []
                    coluna_anterior = None

                    # # Considerando
                    # dx = 0
                    # dy = 0.5
                    # d_teta = 0

                    dx = 0
                    dy = distancia_entre_fotos
                    d_teta = 0

                    print("\n")
                    start_time = time.time()
                    while True:
                        ret, frame = camera.read()
                        if ret:
                            with stream and device:
                                frame = cp.flip(frame, 1)
                                imagem = retifica_imagem(
                                    frame, uimg, vimg, stream)
                                imagem = cp.flip(imagem, 1)

                                imagem_resultado = modelo_yolo.predict(source=imagem, 
                                                                       imgsz=(altura_yolo, largura_yolo), 
                                                                       conf=conf_yolo, 
                                                                       device=0, 
                                                                       classes=classes_yolo, 
                                                                       max_det=maxima_det)
                                imagem_saida = imagem_resultado[0].plot(labels=True, 
                                                                        font_size=3, 
                                                                        line_width=2)
                                boxes = imagem_resultado[0].boxes.cpu().numpy()
                                coluna_atual = cp.array(boxes.xywh.astype(int)[:, 0])
                                linha_atual = cp.array(boxes.xywh.astype(int)[:, 1])
                                largura_atual = cp.array(boxes.xywh.astype(int)[:, 2])
                                altura_atual = cp.array(
                                    boxes.xywh.astype(int)[:, 3])
                                if coluna_anterior is not None:
                                    coluna_anterior = coluna_anterior * 2 * math.pi / largura_transf
                                    coluna_atual_teta = (
                                        coluna_atual * 2 * math.pi / largura_transf) + d_teta
                                    m1 = coluna_anterior.shape[0]
                                    m2 = coluna_atual_teta.shape[0]
                                    m_max = max(m1, m2)
                                    m_min = min(m1, m2)
                                    zeros = cp.zeros((m_max))
                                    if m1 > m2:
                                        zeros[:coluna_atual_teta.shape[0]
                                              ] = coluna_atual_teta
                                        coluna_atual_teta = zeros
                                    elif m2 > m1:
                                        zeros[:coluna_anterior.shape[0]
                                              ] = coluna_anterior
                                        coluna_anterior = zeros
                                    matriz1 = cp.repeat(
                                        coluna_anterior.reshape(1, -1), m_max, axis=0)
                                    matriz2 = cp.repeat(
                                        coluna_atual_teta.reshape(1, -1), m_max, axis=0)
                                    delta_teta = cp.abs(matriz1 - matriz2.T)
                                    indices = cp.argmin(delta_teta, axis=1)
                                    indices = indices[:m_min]
                                    coluna_anterior = coluna_anterior[:m_min]
                                    coluna_atual_teta_aju = coluna_atual_teta[:m_min]
                                    coluna_anterior = coluna_anterior[indices]
                                    largura_atual = largura_atual[:m_min]
                                    coluna_atual_img = coluna_atual[:m_min]
                                    X_p = (dx * cp.tan(coluna_atual_teta_aju) - dy) / \
                                        (cp.tan(coluna_atual_teta_aju) -
                                         cp.tan(coluna_anterior))
                                    Y_p = X_p * cp.tan(coluna_anterior)
                                    distancia = cp.round(
                                        cp.sqrt(cp.power(X_p, 2) + cp.power(Y_p, 2)), 2)

                                    for i in range(len(distancia)):
                                        largura_texto = int(
                                            coluna_atual_img[i] - (largura_atual[i]/2))
                                        altura_texto = int(
                                            linha_atual[i] + (altura_atual[i]/2) + 20)
                                        imagem_saida = cv2.putText(imagem_saida, str(distancia[i]) + "m", (largura_texto, altura_texto),
                                                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                                        imagem_saida = cv2.putText(imagem_saida, str(distancia[i]) + "m", (largura_texto, altura_texto),
                                                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                                coluna_anterior = coluna_atual

                                cv2.imshow("Video Original", frame)
                                cv2.imshow("Video YOLO", imagem_saida)
                                if opcao_gravar_loc:
                                    video_original_loc.append(frame)
                                    video_transformado_loc.append(imagem)
                                    video_transf_yolo_loc.append(imagem_saida)
                            stream.synchronize()
                            fps += 1
                            if cv2.waitKey(1) == ord("q"):
                                break
                        else:
                            print("\nCâmera parou de funcionar...")
                            break
                    end_time = time.time()
                    camera.release()
                    cv2.destroyAllWindows()
                    fps_video = fps/(end_time - start_time)
                    print(f"\nFrames por segundo: {fps_video:.2f}")
                    fps_video = math.ceil(fps_video)
                    print("\nVídeo encerrado com sucesso!")

                    if opcao_gravar_loc:
                        opcao_continua = input(
                            "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")
                        print(
                            "\nCriando vídeos...\n\nAVISO: Isso pode demorar alguns minutos!")
                        time.sleep(3)
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        caminho_completo_gravar_orig = os.path.join(
                            caminho_cam_orig_video, f"{hoje}_video-original-localizacao.avi")
                        caminho_completo_gravar_orig = obter_nome_unico(
                            caminho_completo_gravar_orig)
                        altura_video_original = video_original_loc[0].shape[0]
                        largura_video_original = video_original_loc[0].shape[1]
                        video_saida = cv2.VideoWriter(
                            caminho_completo_gravar_orig, fourcc, fps_video, (largura_video_original, altura_video_original))
                        contador = 1
                        for frame in video_original_loc:
                            frame_video_original = frame.astype("uint8")
                            video_saida.write(frame_video_original)
                            progresso_video = int(
                                (100/len(video_original_loc))*contador)
                            contador += 1
                            os.system("cls")
                            print(f"Vídeo original carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                                100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_original_loc)}]")
                        video_saida.release()
                        print("Vídeo original criado!")
                        opcao_continua = input(
                            "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                        caminho_completo_gravar_transf = os.path.join(
                            caminho_cam_transf_video, f"{hoje}_video-transformado-localizacao.avi")
                        caminho_completo_gravar_transf = obter_nome_unico(
                            caminho_completo_gravar_transf)
                        altura_video_transf = video_transformado_loc[0].shape[0]
                        largura_video_transf = video_transformado_loc[0].shape[1]
                        video_saida = cv2.VideoWriter(
                            caminho_completo_gravar_transf, fourcc, fps_video, (largura_video_transf, altura_video_transf))
                        contador = 1
                        for frame in video_transformado_loc:
                            frame_video_transf = frame.astype("uint8")
                            video_saida.write(frame_video_transf)
                            progresso_video = int(
                                (100/len(video_transformado_loc))*contador)
                            contador += 1
                            os.system("cls")
                            print(f"Vídeo retificado carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                                100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_transformado_loc)}]")
                        video_saida.release()
                        print("Vídeo retificado criado!")
                        opcao_continua = input(
                            "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                        caminho_completo_gravar_yolo = os.path.join(
                            caminho_yolo_video, f"{hoje}_video-transformado-localizacao_{escolha_modelo_yolo}.avi")
                        caminho_completo_gravar_yolo = obter_nome_unico(
                            caminho_completo_gravar_yolo)
                        altura_video_yolo = video_transf_yolo_loc[0].shape[0]
                        largura_video_yolo = video_transf_yolo_loc[0].shape[1]
                        video_saida = cv2.VideoWriter(
                            caminho_completo_gravar_yolo, fourcc, fps_video, (largura_video_transf, altura_video_transf))
                        contador = 1
                        for frame in video_transf_yolo_loc:
                            frame_video_yolo = frame.astype("uint8")
                            video_saida.write(frame_video_yolo)
                            progresso_video = int(
                                (100/len(video_transf_yolo_loc))*contador)
                            contador += 1
                            os.system("cls")
                            print(f"Vídeo com detecção da YOLO carregando...\n{progresso_video:4}% |" + "█"*progresso_video + " "*(
                                100-progresso_video) + f"| Frames carregados: [{contador-1}/{len(video_transf_yolo_loc)}]")
                        video_saida.release()
                        print("Vídeo com detecção da YOLO criado!")
                        opcao_continua = input(
                            "\nAVISO: Os dados no CLI serão apagados!\n\nPressione Enter para continuar\n")

                        os.system("cls")
                        print(
                            f"Vídeo original salvo com sucesso:\n {caminho_completo_gravar_orig}")
                        print(
                            f"\nVídeo retificado salvo com sucesso:\n {caminho_completo_gravar_transf}")
                        print(
                            f"\nVídeo com detecção da YOLO salvo com sucesso:\n {caminho_completo_gravar_yolo}")

                    print("\nVoltando...")

                case 1:
                    documentos = [arquivo for arquivo in os.listdir(caminho_cam_transf_foto) if os.path.isfile(
                        os.path.join(caminho_cam_transf_foto, arquivo))]
                    documentos.sort()
                    print("\nImagens presente na pasta para posição 1:")
                    i = 1
                    for arquivo in documentos:
                        print(f"{i:>2} - {arquivo}")
                        i += 1
                    opcao_foto_loc_1 = input(
                        "Digite o número da imagem que deseja para posição 1: ")
                    while not (opcao_foto_loc_1.isdigit()) or ((len(opcao_foto_loc_1) > 2) and (len(opcao_foto_loc_1) < 0)) or (int(opcao_foto_loc_1) not in (range(1, len(documentos)+1))):
                        print(
                            f"\nSua resposta ({opcao_foto_loc_1}) não corresponde...")
                        print("Imagens presente na pasta:")
                        i = 1
                        for arquivo in documentos:
                            print(f"{i:>2} - {arquivo}")
                            i += 1
                        opcao_foto_loc_1 = input(
                            "Digite o número da imagem que deseja para posição 1: ")

                    print(
                        f"\nFoto selecionada para posição 1: {documentos[int(opcao_foto_loc_1) - 1]}")

                    pos_1_x = float(
                        input(f"\nDigite a posição X da imagem 1 [cm]: ")) / 100
                    pos_1_y = float(
                        input(f"\nDigite a posição Y da imagem 1 [cm]: ")) / 100
                    pos_1_teta = float(
                        input(f"\nDigite o ângulo da imagem 1 [graus]: "))

                    print(
                        f"\nPosição da imagem 1:\nEixo X: {pos_1_x:>5.2f} [m]\nEixo Y: {pos_1_y:>5.2f} [m]\nÂngulo: {pos_1_teta:>5.2f} [graus]")

                    documentos_opcao_2 = documentos.copy()
                    documentos_opcao_2.remove(
                        documentos[int(opcao_foto_loc_1)-1])
                    documentos_opcao_2.sort()

                    print("\nImagens presente na pasta para posição 2:")
                    i = 1
                    for arquivo in documentos_opcao_2:
                        print(f"{i} - {arquivo}")
                        i += 1
                    opcao_foto_loc_2 = input(
                        "Digite o número da imagem que deseja para a posição 2: ")
                    while not (opcao_foto_loc_2.isdigit()) or ((len(opcao_foto_loc_2) > 2) and len(opcao_foto_loc_2) < 0) or (int(opcao_foto_loc_2) not in (range(1, len(documentos_opcao_2)+1))):
                        print(
                            f"\nSua resposta ({opcao_foto_loc_2}) não corresponde...")
                        print("\nImagens presente na pasta para posição 2:")
                        i = 1
                        for arquivo in documentos_opcao_2:
                            print(f"{i} - {arquivo}")
                            i += 1
                        opcao_foto_loc_2 = input(
                            "Digite o número da imagem que deseja para a posição 2: ")

                    print(
                        f"\nImagem selecionada para posição 2: {documentos_opcao_2[int(opcao_foto_loc_2) - 1]}")

                    pos_2_x = float(
                        input(f"\nDigite a posição X da imagem 2 [cm]: ")) / 100
                    pos_2_y = float(
                        input(f"\nDigite a posição Y da imagem 2 [cm]: ")) / 100
                    pos_2_teta = float(
                        input(f"\nDigite o ângulo da imagem 2 [graus]: "))

                    print(
                        f"\nPosição da imagem 2:\nEixo X: {pos_2_x:>5.2f} [m]\nEixo Y: {pos_2_y:>5.2f} [m]\nÂngulo: {pos_2_teta:>5.2f} [graus]")

                    arquivo_imagem_1 = documentos[int(opcao_foto_loc_1) - 1]
                    imagem_loc_1 = cv2.imread(
                        f"{caminho_cam_transf_foto}\{arquivo_imagem_1}")
                    arquivo_imagem_2 = documentos_opcao_2[int(
                        opcao_foto_loc_2) - 1]
                    imagem_loc_2 = cv2.imread(
                        f"{caminho_cam_transf_foto}\{arquivo_imagem_2}")

                    print("\nDetectando objetos na imagem...")

                    imagem_resultado_loc_1 = modelo_yolo.predict(source=imagem_loc_1, imgsz=(
                        altura_yolo, largura_yolo), conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
                    imagem_saida_loc_1 = imagem_resultado_loc_1[0].plot(
                        labels=False, font_size=3, line_width=2)
                    boxes_1 = imagem_resultado_loc_1[0].boxes.cpu().numpy()

                    imagem_resultado_loc_2 = modelo_yolo.predict(source=imagem_loc_2, imgsz=(
                        altura_yolo, largura_yolo), conf=conf_yolo, device=0, classes=classes_yolo, max_det=maxima_det)
                    imagem_saida_loc_2 = imagem_resultado_loc_2[0].plot(
                        labels=False, font_size=3, line_width=2)
                    boxes_2 = imagem_resultado_loc_2[0].boxes.cpu().numpy()

                    print("\nCalculando distância dos objetos...")

                    with stream and device:
                        
                        dx = pos_2_x - pos_1_x
                        dy = pos_2_y - pos_1_y
                        d_teta = pos_2_teta - pos_1_teta
                        d_teta = d_teta * math.pi / 180

                        print(f"\nDelta X: {dx:.2f} [m]\tDelta Y: {dy:.2f} [m]\tDelta θ: {d_teta:.2f} [rad]")

                        vect_teta_1 = cp.array(boxes_1.xywh.astype(int)[:, 0])
                        print(f"\nVetor de colunas da imagem 1:\n{vect_teta_1}")
                        vect_teta_1 = vect_teta_1 * 2 * math.pi / largura_transf
                        m_1 = vect_teta_1.shape[0]

                        print(f"\nVetor de ângulos [rad] da imagem 1:\n{vect_teta_1}\n\nComprimento vetor 1: {m_1}")

                        vect_altura_2 = cp.array(boxes_2.xywh.astype(int)[:, 3])
                        vect_largura_2 = cp.array(boxes_2.xywh.astype(int)[:, 2])
                        vect_linha_2 = cp.array(boxes_2.xywh.astype(int)[:, 1])
                        vect_teta_2_pixel = cp.array(boxes_2.xywh.astype(int)[:, 0])
                        vect_teta_2 = (vect_teta_2_pixel * 2 * math.pi / largura_transf) - d_teta
                        m_2 = vect_teta_2.shape[0]

                        print(f"\nVetor de colunas da imagem 2:\n{vect_teta_2_pixel}\n\nVetor de ângulos [rad] da imagem 2 corrigido por alpha:\n{vect_teta_2}\n\nComprimento vetor 2: {m_2}")

                        m_max = max(m_1, m_2)
                        m_min = min(m_1, m_2)
                        zeros = cp.zeros((m_max))

                        print(f"\nTamanho do maior vetor: {m_max}\nTamanho do menor vetor: {m_min}")

                        if m_1 > m_2:
                            zeros[:vect_teta_2.shape[0]] = vect_teta_2
                            vect_teta_2 = zeros
                            print(f"\nVetor 1:\n{vect_teta_1}\nVetor 2 completado com zeros:\n{vect_teta_2}")
                        elif m_2 > m_1:
                            zeros[:vect_teta_1.shape[0]] = vect_teta_1
                            vect_teta_1 = zeros
                            print(f"\nVetor 1 completado com zeros:\n{vect_teta_1}\nVetor 2:\n{vect_teta_2}")
                        else:
                            print(f"\nVetor 1:\n{vect_teta_1}\nVetor 2:\n{vect_teta_2}")

                        matriz_1 = cp.repeat(vect_teta_1.reshape(1, -1), m_max, axis=0)
                        matriz_2 = cp.repeat(vect_teta_2.reshape(1, -1), m_max, axis=0)

                        print(f"\nMatriz do vetor 1:\n{matriz_1}\n\nMatriz do vetor 2:\n{matriz_2}")

                        delta_teta = cp.abs(matriz_1 - matriz_2.T)

                        print(f"\nMatriz delta (valores absolutos):\n{delta_teta}")

                        indices = cp.argmin(delta_teta, axis=1)
                        indices = indices[:m_min]
                        vect_teta_1 = vect_teta_1[:m_min]
                        vect_teta_2 = vect_teta_2[:m_min]
                        vect_teta_1 = vect_teta_1[indices]

                        print(f"\nVetor índices:\n{indices}\n\nVetor 1 corrigido com índices:\n{vect_teta_1}\nVetor 2 corrigido com índices:\n{vect_teta_2}")

                        X_p = (dx * cp.tan(vect_teta_2) - dy) / (cp.tan(vect_teta_2) - cp.tan(vect_teta_1))
                        Y_p = X_p * cp.tan(vect_teta_1)

                        print(f"\nVetor de Xp:\n{X_p}\nVetor de Yp:\n{Y_p}")

                        distancia_objs = cp.sqrt(cp.power(X_p, 2) + cp.power(Y_p, 2))
                        distancia_objs = cp.round(distancia_objs, 2)

                        print(f"\nVetor distância:\n{distancia_objs}")

                        for i in range(len(distancia_objs)):
                            largura_texto = int(
                                vect_teta_2_pixel[i] - (vect_largura_2[i]/2))
                            altura_texto = int(
                                vect_linha_2[i] + (vect_altura_2[i]/2) + 20)
                            imagem_saida_loc_2 = cv2.putText(imagem_saida_loc_2, 
                                                             str(distancia_objs[i]) + "m", 
                                                             (largura_texto, altura_texto),
                                                             cv2.FONT_HERSHEY_DUPLEX, 
                                                             0.6, 
                                                             (0, 0, 0), 
                                                             3, 
                                                             cv2.LINE_AA)
                            imagem_saida_loc_2 = cv2.putText(imagem_saida_loc_2, 
                                                             str(distancia_objs[i]) + "m", 
                                                             (largura_texto, altura_texto),
                                                             cv2.FONT_HERSHEY_DUPLEX, 
                                                             0.6, 
                                                             (255, 255, 255), 
                                                             1, 
                                                             cv2.LINE_AA)

                    stream.synchronize()

                    cv2.imshow("Imagem posicao 1", imagem_saida_loc_1)
                    cv2.waitKey(0)
                    cv2.imshow("Imagem posicao 2", imagem_saida_loc_2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    opcao_salva_foto_loc = input(
                        "\nDistâncias calculadas com sucesso. O que deseja fazer:\n0 - Salvar a imagem com as distâncias\n1 - Voltar\nDigite sua escolha (0, 1): ")
                    while not (opcao_salva_foto_loc.isdigit()) or len(opcao_salva_foto_loc) != 1 or (int(opcao_salva_foto_loc)) not in [0, 1]:
                        opcao_salva_foto_loc = input(
                            f"\nSua resposta ({opcao_salva_foto_loc}) não corresponde...\nDistâncias calculadas com sucesso. O que deseja fazer:\n0 - Salvar a imagem com as distâncias\n1 - Voltar\nDigite sua escolha (0, 1): ")
                    opcao_salva_foto_loc = int(opcao_salva_foto_loc)
                    match opcao_salva_foto_loc:

                        case 0:
                            nome_1, extensao_1 = os.path.splitext(
                                f"{arquivo_imagem_1}")
                            nome_2, extensao_2 = os.path.splitext(
                                f"{arquivo_imagem_2}")
                            posicao_1 = nome_1.rfind("_") + 1
                            posicao_2 = nome_2.rfind("_") + 1
                            caminho_completo_foto_dist = os.path.join(
                                caminho_cam_transf_dist, f"{hoje}_localizacao_{nome_1[posicao_1 : ]}-{nome_2[posicao_2 : ]}.jpg")
                            caminho_completo_foto_dist = obter_nome_unico(
                                caminho_completo_foto_dist)
                            cv2.imwrite(caminho_completo_foto_dist, imagem_saida_loc_2, [
                                        cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(
                                f"\nFoto salva com sucesso:\n {caminho_completo_foto_dist}")

                        case 1:
                            print("\nVoltando...")
                            time.sleep(0.3)

            executar_loop = 1

        case 4:
            print("\nSaindo...\n")
            time.sleep(2)
            os.system("cls")
            executar_loop = 0

        case _:
            print("\nEncerrando programa...\n\nComando inválido!")
            time.sleep(2)
            executar_loop = 0
