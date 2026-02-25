# FED-Meta com Proxy-NCA em Aprendizado Federado

## Visão Geral

Método de aprendizado métrico federado baseado em meta-aprendizado no qual a personalização do modelo é realizada por meio da adaptação de *proxies de classe* no espaço latente compartilhado.

Diferentemente de abordagens clássicas de aprendizado federado que realizam *fine-tuning* completo do modelo localmente, o método proposto mantém um espaço de embeddings global e permite que cada cliente personalize apenas sua fronteira de decisão métrica por meio de deslocamentos locais dos proxies.

---

## Modelo Global

O modelo global é composto por:

- Um backbone de extração de embeddings:

$$
f_{\theta}(x) \rightarrow z \in \mathbb{R}^{D}
$$

- Um conjunto de proxies de classe globais:

$$
P \in \mathbb{R}^{C \times D}
$$

onde:

- $C$ representa o número de classes  
- $D$ representa a dimensão do espaço latente  

A classificação é realizada via aprendizado métrico utilizando uma função de perda baseada em Proxy-NCA:

$$
\ell(z, y) =
\text{CE}
\left(
\frac{z^{\top}P_y}{T}
\right)
$$

onde:

- $z = f_{\theta}(x)$  
- $P_y$ é o proxy associado à classe $y$  
- $T$ é um parâmetro de temperatura  

---

## Personalização Local via Offsets de Proxy

Cada cliente $k$ adapta localmente os proxies globais por meio de um deslocamento específico:

$$
P^{(k)} = P + \Delta P^{(k)}
$$

onde:

- $P$ são os proxies globais compartilhados  
- $\Delta P^{(k)}$ é o offset local do cliente  

A personalização ocorre apenas no espaço métrico, preservando a estrutura global do backbone.

---

## Treinamento como Meta-Aprendizado Federado

Cada rodada federada é interpretada como uma tarefa de adaptação local composta por dois estágios:

### Inner Loop – Adaptação Local

Dado um conjunto de suporte $S_k$, o cliente resolve:

$$
\min_{\theta', \Delta P}
\mathcal{L}_{sup}
\left(
f_{\theta'}(x),
P + \Delta P
\right)
+
\lambda \|\Delta P\|^2
$$

onde:

- $\theta'$ representa os parâmetros adaptados do backbone  
- $\Delta P$ representa o offset dos proxies  
- $\lambda$ é um termo de regularização  

---

### Outer Loop – Meta-Gradiente

Após a adaptação, os gradientes da perda no conjunto de consulta $Q_k$ são calculados com relação aos parâmetros globais:

$$
\nabla_{\theta, P}
\mathcal{L}_{query}
\left(
f_{\theta'}(x),
P + \Delta P
\right)
$$

Durante esse estágio, o offset $\Delta P$ é tratado como constante:

$$
\tilde{P}^{(k)} = P + \text{stopgrad}(\Delta P)
$$

Os gradientes resultantes são enviados ao servidor.

---

## Agregação no Servidor

O servidor realiza uma agregação ponderada dos gradientes recebidos:

$$
g = \sum_{k \in \mathcal{K}}
\frac{n_k}{\sum_j n_j}
\nabla_k
$$

e atualiza os parâmetros globais utilizando um otimizador Adam:

$$
(\theta, P)
\leftarrow
\text{Adam}
\left(
(\theta, P),
g
\right)
$$

Assim, o treinamento global ocorre por meio de uma otimização meta baseada no desempenho pós-adaptação dos clientes.

---

## Inferência Personalizada

Durante a avaliação, cada cliente:

1. Recebe os parâmetros globais  
2. Adapta localmente o backbone e os proxies utilizando o conjunto de suporte  
3. Realiza predição no conjunto de consulta com:

$$
\hat{y} =
\arg\max_{c}
\left(
\frac{z^{\top}(P_c + \Delta P_c)}{T}
\right)
$$

---




## Pseudo-código do Algoritmo (FED-Meta + Proxy-NCA)

**Entradas:**

- Distribuição de clientes $\mathcal{K}$
- Modelo global com backbone $f_{\theta}$ e proxies $P \in \mathbb{R}^{C \times D}$
- Hiperparâmetros: $\alpha$ (inner LR), $\beta$ (outer LR), $\lambda$ (reg. offset), $T$ (temperatura), $E$ (inner epochs)
- Perda Proxy-NCA:

$$
\mathcal{L}_{\text{PNCA}}(z, P, y) =
\text{CE}
\left(
\frac{zP^{\top}}{T}, y
\right)
$$

---

### Algoritmo 1 — Treinamento Federado PED-Meta com Proxy-NCA

**No Servidor (rodada $t = 1,\dots,R$):**

1. Amostrar um subconjunto de clientes
$$
K_t \subseteq \mathcal{K}
$$

2. Enviar
$$
(\theta_t, P_t)
$$
para cada cliente $k \in K_t$

3. Receber de cada cliente $k$ os gradientes
$$
\nabla_{\theta,P}
\mathcal{L}_{\text{query}}^{(k)}
$$
e o peso $n_k$

4. Agregar gradientes por média ponderada:

$$
g_t =
\sum_{k \in K_t}
\frac{n_k}{\sum_j n_j}
\nabla_{\theta,P}
\mathcal{L}_{\text{query}}^{(k)}
$$

5. Atualizar parâmetros globais:

$$
(\theta_{t+1}, P_{t+1})
\leftarrow
\text{Adam}
\left(
(\theta_t, P_t),
g_t, \beta
\right)
$$

6. Repetir até $R$ rodadas

---

### Algoritmo 2 — Rotina do Cliente $k$ (rodada $t$)

**Entrada:** $(\theta_t, P_t)$, dados locais divididos em *support* $S_k$ e *query* $Q_k$

1. Inicializar cópia adaptável do backbone:

$$
\theta' \leftarrow \theta_t
$$

2. Inicializar offsets dos proxies:

$$
\Delta P \leftarrow 0
\quad
\left(
\text{mesma forma de } P_t
\right)
$$

---

#### (A) Inner Loop — Adaptação em $S_k$

Para época $e = 1,\dots,E$

Para minibatch $(x,y) \sim S_k$

Calcular embeddings:

$$
z \leftarrow f_{\theta'}(x)
$$

Proxies efetivos:

$$
P^{(k)} \leftarrow P_t + \Delta P
$$

Perda com regularização:

$$
\mathcal{L}_{\text{sup}}
\leftarrow
\mathcal{L}_{\text{PNCA}}
\left(
z,
P^{(k)},
y
\right)
+
\lambda
\left\|
\Delta P
\right\|^2
$$

Atualizar backbone adaptado:

$$
\theta'
\leftarrow
\theta'
-
\alpha
\nabla_{\theta'}
\mathcal{L}_{\text{sup}}
$$

Atualizar offset:

$$
\Delta P
\leftarrow
\Delta P
-
\alpha
\nabla_{\Delta P}
\mathcal{L}_{\text{sup}}
$$

---

#### (B) Outer Loop — Gradientes meta em $Q_k$

Definir proxies efetivos sem gradiente no offset:

$$
\tilde{P}^{(k)}
\leftarrow
P_t
+
\text{stopgrad}
\left(
\Delta P
\right)
$$

Inicializar acumuladores de gradiente:

$$
G_{\theta} \leftarrow 0
\qquad
G_{P} \leftarrow 0
$$

Para minibatch $(x,y) \sim Q_k$

$$
z \leftarrow f_{\theta'}(x)
$$

$$
\mathcal{L}_{\text{query}}
\leftarrow
\mathcal{L}_{\text{PNCA}}
\left(
z,
\tilde{P}^{(k)},
y
\right)
$$

Acumular gradientes:

$$
G_{\theta}
\leftarrow
G_{\theta}
+
\nabla_{\theta_t}
\mathcal{L}_{\text{query}}
$$

$$
G_{P}
\leftarrow
G_{P}
+
\nabla_{P_t}
\mathcal{L}_{\text{query}}
$$

8. Retornar ao servidor:

- Gradientes $(G_{\theta}, G_{P})$ (normalizados por nº de amostras do query)
- Peso

$$
n_k = |Q_k|
$$

---

### Observações Importantes

- **Personalização** ocorre via $\Delta P$ (offset dos proxies), mantendo $P_t$ como referência global.
- O servidor **não** aprende $\Delta P$: ele é descartado a cada rodada
$$
\text{stopgrad}
\left(
\Delta P
\right)
$$
evitando contaminar o modelo global com deslocamentos geométricooss.
- O backbone pode ser adaptado no *inner loop* para melhor alinhar o embedding ao domínio do cliente, mas a atualização global é feita via meta-gradientes do *query*.



