def uncertainty_guided_pipeline(problem, problem_type, model_client):
    """
    Основной пайплайн с uncertainty-guided reasoning
    """
    context = ""
    final_answer = None
    reasoning_steps = []
    
    while not is_complete(context, problem_type):
        # 1. Измеряем uncertainty в текущем состоянии
        uncertainty = compute_uncertainty(context, problem, model_client)
        
        # 2. Принимаем решение о стратегии
        if uncertainty > get_threshold(problem_type):
            # Высокая uncertainty → CoT reasoning
            next_step = cot_decoding(context, problem, model_client, k=5)
        else:
            # Низкая uncertainty → прямая генерация
            next_step = greedy_generation(context, problem, model_client)
        
        context += next_step
        reasoning_steps.append({
            'step': next_step,
            'uncertainty': uncertainty,
            'used_cot': uncertainty > get_threshold(problem_type)
        })
    
    return extract_final_answer(context), reasoning_steps

def get_uncertainty_points_math(context):
    """
    Определяем где измерять uncertainty в математических задачах
    """
    trigger_patterns = [
        r"Let's.*step by step",           # Начало рассуждения
        r"So we have",                    # Промежуточные выводы  
        r"Therefore",                     # Логические переходы
        r"Next,",                         # Следующий шаг
        r"The answer is",                 # Финальный ответ
        r"\d+\.?\d*\s*[+\-*/=]",        # Перед вычислениями
        r"Step \d+:",                     # Явные шаги
    ]
    
    # Проверяем, находимся ли мы в одной из критических точек
    for pattern in trigger_patterns:
        if re.search(pattern, context[-50:]):  # Последние 50 символов
            return True
    return False

def get_uncertainty_points_qa(context, passages):
    """
    Определяем где измерять uncertainty в QA задачах
    """
    trigger_patterns = [
        r"According to",                  # Ссылка на источник
        r"From the passage",              # Извлечение информации
        r"We can see that",               # Логический вывод
        r"Combining this with",           # Связывание фактов
        r"This means",                    # Интерпретация
        r"The answer is",                 # Финальный ответ
    ]
    
    # Дополнительно: проверяем переходы между источниками
    source_transitions = detect_source_transitions(context, passages)
    
    for pattern in trigger_patterns:
        if re.search(pattern, context[-50:]):
            return True
    
    return source_transitions

def compute_pd_uncertainty(context, problem, model_client):
    """
    Вычисляем P-D uncertainty для следующего токена
    """
    # Формируем prompt для предсказания следующего токена
    prompt = format_prompt(context, problem)
    
    # Получаем топ-20 логитов от модели
    response = model_client.get_logprobs(
        prompt=prompt,
        max_tokens=1,  # Только следующий токен
        top_logprobs=20
    )
    
    # Извлекаем топ-2 вероятности
    top_probs = response.choices[0].logprobs.top_logprobs[0]
    probs_list = [math.exp(logprob.logprob) for logprob in top_probs.values()]
    probs_list.sort(reverse=True)
    
    p1, p2 = probs_list[0], probs_list[1]
    
    # P-D uncertainty
    uncertainty = 1 - (p1 - p2)
    
    return {
        'uncertainty': uncertainty,
        'top_token': list(top_probs.keys())[0],
        'p1': p1,
        'p2': p2,
        'confidence_gap': p1 - p2
    }

def cot_decoding(context, problem, model_client, k=5):
    """
    CoT decoding с выбором наиболее уверенного пути
    """
    reasoning_paths = []
    
    # Генерируем k путей рассуждения
    for i in range(k):
        path = generate_reasoning_step(
            context=context,
            problem=problem,
            model_client=model_client,
            temperature=0.7,  # Для разнообразия
            max_tokens=100
        )
        
        # Вычисляем confidence для этого пути
        confidence = compute_path_confidence(path, model_client)
        
        reasoning_paths.append({
            'text': path,
            'confidence': confidence,
            'path_id': i
        })
    
    # Выбираем путь с наибольшей confidence
    best_path = max(reasoning_paths, key=lambda x: x['confidence'])
    
    return best_path['text']

def compute_path_confidence(text, model_client):
    """
    Вычисляем confidence для сгенерированного текста
    """
    # Tokenize текст
    tokens = model_client.tokenize(text)
    
    total_confidence = 0
    for i, token in enumerate(tokens):
        # Получаем вероятности для каждого токена в контексте
        context_tokens = tokens[:i]
        token_probs = get_token_probabilities(context_tokens, model_client)
        
        if token in token_probs:
            token_confidence = token_probs[token]
            total_confidence += math.log(token_confidence)
    
    # Нормализуем по длине
    return total_confidence / len(tokens) if len(tokens) > 0 else 0

# Пример: "John has 15 apples. He gives away 1/3 of them. How many does he have left?"

# Шаг 1: Начальный контекст пуст, начинаем рассуждение
context = ""
uncertainty = compute_pd_uncertainty(context, problem, model) 
# uncertainty = 0.65 > threshold(0.4) → используем CoT

# Шаг 2: CoT генерирует несколько путей:
# Path 1: "Let's solve this step by step. John starts with 15 apples..."
# Path 2: "First, I need to find 1/3 of 15 apples..."  
# Path 3: "To solve this, let me calculate how many apples John gives away..."
# Выбираем путь с максимальной confidence

# Шаг 3: Продолжаем с выбранным контекстом
context = "Let's solve this step by step. John starts with 15 apples. He gives away 1/3 of them."
uncertainty = compute_pd_uncertainty(context, problem, model)
# uncertainty = 0.25 < threshold(0.4) → используем greedy generation

# Результат: "So he gives away 15 ÷ 3 = 5 apples. Therefore he has 15 - 5 = 10 apples left."

# Пример HotpotQA с двумя источниками информации

# Шаг 1: Анализ первого источника
context = "From the first passage, I can see that..."
uncertainty = compute_pd_uncertainty(context, problem, model)
# uncertainty = 0.35 < threshold(0.45) → greedy generation

# Шаг 2: Переход ко второму источнику (триггер uncertainty)  
context += "Looking at the second passage..."
uncertainty = compute_pd_uncertainty(context, problem, model)
# uncertainty = 0.55 > threshold(0.45) → CoT reasoning

# Шаг 3: Связывание информации (сложная часть)
# Генерируем k=5 способов связать факты, выбираем лучший