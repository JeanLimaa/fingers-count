from mediapipe import solutions
import cv2

# Inicializando a captura de vídeo da webcam
frame = cv2.VideoCapture(0)

# Inicializando o modelo de detecção de mãos da MediaPipe
hand = solutions.hands
Hand = hand.Hands(max_num_hands=1)
draw = solutions.drawing_utils

# Definindo uma variável global para verificar se o programa deve ser fechado
exit_program = False


def get_button_coordinates(width, height):
    global exit_button_x, exit_button_y, exit_button_radius

    # Definindo as coordenadas do botão "Sair" no canto inferior esquerdo
    exit_button_x, exit_button_y, exit_button_radius = int(0.9 * width), int(0.9 * height), 30


# Função que verifica se o dedo está sobre o botão de sair
def is_hand_over_exit_button(hand_landmarks, width, height):
    if hand_landmarks is not None:
        # Obtendo as coordenadas do dedo indicador (ponta)
        indicator_tip_x = int(hand_landmarks[8].x * width)
        indicator_tip_y = int(hand_landmarks[8].y * height)

        # Verificando se o dedo indicador está sobre o botão "Sair"
        distance_to_button = ((exit_button_x - indicator_tip_x) ** 2 + (exit_button_y - indicator_tip_y) ** 2) ** 0.5
        return distance_to_button < exit_button_radius

    return False


# Função que exibe o botão de sair - ao clicar nele fechará o programa.
def show_exit_button(img, width, height, point):
    # Coordenadas do texto "Toque para sair"
    text_to_exit_x, text_to_exit_y = int(0.785 * width), int(0.865 * height) - 30

    # Botão de sair
    button_color = (0, 255, 0) if is_hand_over_exit_button(point.landmark, width, height) else (0, 0, 255)
    cv2.circle(img, (exit_button_x, exit_button_y), exit_button_radius, button_color, -1)
    cv2.putText(img, 'Sair', (exit_button_x - 20, exit_button_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

    # Desenhando o texto "Toque para sair" acima do botão
    cv2.putText(img, 'Toque para sair', (text_to_exit_x, text_to_exit_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)


# Função que exibe a contagem dos dedos levantados
def show_count(img, width, count):
    corner_position = (width - 50, 50)
    cv2.circle(img, corner_position, 30, (0, 0, 255), -1)  # Círculo com cor de fundo vermelha
    text_size = cv2.getTextSize(str(count), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 2)[0]
    text_position = (width - 50 - text_size[0] // 2, 50 + text_size[1] // 2)
    cv2.putText(img, str(count), text_position, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)


def generate_frames():
    global exit_program

    while True:
        # Lendo o frame do vídeo
        success, img = frame.read()

        # pegando as dimensoes da imagem
        height, width, _ = img.shape

        if not success:
            break

        # Convertendo o frame para o formato RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processando o frame para obter os resultados da detecção de mãos
        results = Hand.process(img_rgb)
        hands_points = results.multi_hand_landmarks

        points_list = []
        count = 0

        if hands_points:
            # Iterando sobre cada mão detectada no frame
            for point in hands_points:
                # Desenhando as landmarks e conexões da mão no frame
                draw.draw_landmarks(img, point, hand.HAND_CONNECTIONS)

                # Iterando sobre cada landmark da mão
                for _, cord in enumerate(point.landmark):
                    cord_x, cord_y = int(cord.x * width), int(cord.y * height)
                    points_list.append((cord_x, cord_y))
                    cv2.putText(img, str(_), (cord_x, cord_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Identificando a direção da mão (se está para direita ou esquerda)
                right_hand = points_list[1][0] > points_list[0][0]

                # Identificando se o polegar está levantado
                # Obs: a lógica dos demais dedos é diferente do polegar, inclusive a relação da direção da mão.
                if right_hand:
                    thumb_raised = points_list[4][0] > points_list[3][0]
                else:
                    thumb_raised = points_list[4][0] < points_list[3][0]

                # Verificando se o polegar está levantado
                if thumb_raised:
                    count += 1

                # Identificando se os demais dedos estão levantados
                fingers = [8, 12, 16, 20]
                for x in fingers:
                    if points_list[x][1] < points_list[x - 2][1]:
                        count += 1

                # Obtendo as cordenadas que o botão terá
                get_button_coordinates(width, height)

                # Exibindo a contagem no canto superior direito
                show_count(img, width, count)

                # Exibir o botão de sair
                show_exit_button(img, width, height, point)

                # Verificando se a mão está sobre o botão "Sair" e definindo a variável de saída
                exit_program = is_hand_over_exit_button(point.landmark, width, height)

        # Exibindo o frame resultante
        cv2.imshow('Imagem', img)
        cv2.waitKey(1)

        # Verificando se o programa deve ser fechado
        if exit_program:
            break

    # Liberando o objeto de captura de vídeo e fechando todas as janelas
    frame.release()
    cv2.destroyAllWindows()


# Chamando a função para gerar os frames
generate_frames()
