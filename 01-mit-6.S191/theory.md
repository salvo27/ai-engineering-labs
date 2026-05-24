# Il Contesto Storico: L'Era della Generative AI ed Edge AI
Nel **2022**, il rilascio ufficiale al pubblico di **GPT3** ha segnato un punto di non ritorno: il modo in cui accediamo alle informazioni e percepiamo la tecnologia è cambiato drasticamente. Processi aziendali e di sviluppo che prima richiedevano giorni, oggi vengono completati in pochi minuti.

La velocità di questa evoluzione è fulminea. Appena un anno dopo, nel **2023**, il rilascio di **GPT4** ha ridefinito i limiti della comprensione linguistica dei modelli. Tuttavia, queste enormi architetture condividevano un vincolo strutturale: la totale **dipendenza dal Cloud**.

## Il Cambio di Paradigma: Dal Cloud al Locale (Edge AI)
La vera rivoluzione ingegneristica si sta compiendo oggi. Modelli con benchmark simili o superiori a quelli di GPT4 non sono più confinati all'interno di data center mastodontici.

Grazie ad architetture innovative, come quelle sviluppate da **Liquid AI**, e a tecniche avanzate di ottimizzazione, i Large Language Models (LLM) oggi possono vivere **completamente in locale**. Possono girare offline, senza connessione internet, mentre sei in aereo o all'interno di una galleria, direttamente sulla RAM di dispositivi consumer.

Questo manuale nasce per esplorare la tecnologia alla base di questo salto quantico: il **Deep Learning**.

# Cos'è il Deep Learning e perché proprio ora?
Il Deep Learning è un sottocampo del Machine Learning che, a sua volta fa parte dell'intelligenza artificiale. La sua caratteristica principale è che vengono estratti dei pattern dai dati grezzi, attraverso l'uso di reti neurali artificiali.

Perché proprio ora sta esplodendo? Per tre fattori:
- Big data: oggi abbiamo a disposizione datasets massicci;
- Hardware: abbiamo raggiunto livelli tali da supportare il DL, ad esempio GPU altamente parallelizzabili;
- Software: gli algoritmi sono migliorati in maniera notevole, abbiamo toolbox come TensorFlow e PyTorch (che utilizzeremo all'interno del manuale).

## Il Perceptron (Percettone), l'atomo della rete neurale.
Il perceptron è il blocco costruttivo fondamentale, il tipo di rete più semplice possibile. Funziona come un neurone biologico semplificato, che riceve degli input, li elabora, e produce un output.

L'output di un perceptron è definito dalla seguente formula matematica:

$$y = g \left( w_0 + \sum_{i=1}^{m} x_i w_i \right)$$

che in notazione vettoriale diventa:

$$y = g \left(w_0 + X^T W \right)$$

Cosa significano questi termini? Spieghiamo la formula:

- X = [x_1,...,x_m]^T : è il vettore degli imput;

- W = [w_1,...,wm]^T : è il vettore dei pesi, servono a determinare che importanza dare ad ogni input;

- w_0 : è il bias, permette di translare la funzione di attivazione per adattarsi meglio ai dati. Immaginiamo il Perceptron come un buttafuori all'ingresso di un locale con determinate regole di abbigliamento. Gli input sono i requisiti del cliente (ad esempio quanto è vestito bene e quanti anni ha). 
Un bias molto alto significa che il buttafuori è estremamente permissivo, ad esempio se il locale è vuoto fa entrare quasi tutti. 
Un bias molto basso significa che il buttafuori è severissimo, il locale magali è frequentatissimo, e lui ha degli standard molto alti, fa entrare solo i clienti che hanno degli input eccezionali.

Dal punto di vista matematico, se non avessimo il bias, la retta sarebbe sempre costretta a passare per l'origine degli assi, quindi a vedere o bianco o nero, senza alcuna sfumatura di grigio.

Ad esempio, se dovessimo distinguere le persone malate da quelle sane, senza bias, probabilmente andremmo ad inserire un ragazzo raffreddato dentro il gruppo delle persone malate, senza alcuna minima tolleranza;

- g : ovvero la funzione di attivazione non lineare, sotto entreremo nel dettaglio;

- y : l'output predetto.

## L'importanza della non linearità
Senza la funzione di attivazione g, la rete sarebbe soltanto una combinazione lineare di input, incapace di modellare problemi complessi. 

Se colleghiamo in cascata un secondo strato di neuroni, questo prenderà in input una combinazione lineare, e darà in output un altra combinazione lineare. Da un punto di vista puramente matematico, la combinazione lineare di una combinazione lineare, è essa stessa una combinazione lineare. Questo significherebbe che se anche riuscissimo a costruire una rete neurale gigantesca, con 100 strati e miliardi di parametri, senza funzioni di attivazione non lineari, quella rete si comporterà matematicamente come il semplicissimo perceptron. Avremmo costruito un castello gigantesco ma potente quanto una piccola casetta.

Immaginiamo di dover addestrare un modello per distinguere le cellule sane da quelle tumorali. Nel grafico dei dati, scopriamo che le cellule tumorali si concentrano tutte al centro, formando una bolla (un cerchio), mentre le cellule sane le circondano.

**Senza la funzione di attivazione $g$ (Solo Lineare):** Il modello ha a disposizione solo "linee rette". Può tracciare una retta in diagonale, in verticale o in orizzontale. Ma come isoliamo un cerchio al centro usando solo una linea retta? È impossibile. Saremmo costretti a tagliare sempre il dataset a metà, accumulando un'infinità di errori. Il modello non ha la capacità di "curvare" la sua decisione.

**Con la funzione di attivazione $g$ (Non Lineare):** Introducendo funzioni come la ReLU o la Sigmoide, che vedremo, permettiamo alla matematica del modello di piegarsi. La rete può iniziare a combinare più rette curvate fino a creare un perimetro perfetto attorno alle cellule tumorali.

Le funzioni di attivazione più comuni sono:

- Sigmoide: $$g(z) = \frac{1}{1 + e^{-z}}$$, utile per le probabilità (output tra 0 e 1), ad esempio la probabilità di essere promosso o bocciato ad un esame. Si usa nell'output layer, ovvero lo strato finale, quando il compito della rete diventa quello di classificare (non la usiamo negli hidden layers a causa del Vanishing Gradient, che è un problema che vedremo successivamente);

- ReLU (rectified linear unit): $$g(z) = \max(0, z)$$. La classica rampa, è la più usata per la sua semplicità computazionale negli hidden layers, che a breve vedremo. Il segreto del suo utilizzo è la semplicità computazionale, per una GPU è un semplice if, questo riduce drasticamente l'addestramento di un modello;

- Tanh: $$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$, è molto simile alla sigmoide, ma mappa l'output tra valori compresi tra -1 ed 1. Avendo l'output centrato sullo zero, rende l'addestramento dello stato successivo più stabile e veloce.

# Architettura delle reti neurali: gli strati
Per risolvere problemi complessi, i perceptron vengono utilizzati in strati (layers):

- Input Layers: i dati d'ingresso;

- Hidden Layers: Strati intermedi che estraggono feature sempre più astratte. Se ci sono moltri strati, la rete si dice "Deep";

- Output layers: forniscono la decisione finale, la classificazione.

In un Dense Layer, che sarebbe uno strato completamente connesso, ogni unità riceve input da tutte le unità dello strato precedente.
Questo significa che lo strato esegue una mappatura globale delle informazioni che lo precedono. Ciascun neurone applicherà il proprio set di pesi a tutti gli input ricevuti, sommerà il proprio bias e farà passare il risultato attraverso la funzione di attivazione.

$$z_i = w_{0,i} + \sum_{j=1}^{m} x_j w_{j,i}$$

Dove $$w_{j,i}$$ è il peso che collega l'input j al neurone i.

## Single Layer Neural Network
Una Single Layer Neural Network (SLNN) (o spesso chiamata semplicemente Single Layer Perceptron Network) non è altro che una rete neurale in cui è presente un unico Dense Layer. Ovvero il layer di output. (Nessun hidden layer)

$$\hat{y}_i = g\!\left(\mathbf{w}_i^\top \mathbf{x} + b_i\right)$$

In forma vettoriale (tutti gli output insieme):

$$\hat{\mathbf{y}} = g\!\left(W\mathbf{x} + \mathbf{b}\right)$$

## Multi Layer Perceptron (MLP)
La MLP è una rete neurale composta da più strati densi (almeno 2), ed è composta da almeno un hidden layer. E' l'architettura classica in cui l'informazione scorre in un'unica direzione, dall'input all'output. 

## Deep Neural Network
Una MLP può essere una Deep Network se ha molti strati nascosti.

$$z_{k,i} = w_{0,i}^{(k)} + \sum_{j=1}^{n_{k-1}} g(z_{k-1,j})\, w_{j,i}^{(k)}$$

dove:
- $k$ : indice del layer corrente;

- $i$ : indice del neurone corrente nel layer $k$;

- $n_{k-1}$ : numero di neuroni nel layer precedente;

- $g(z_{k-1,j})$ : attivazione del neurone $j$ del layer precedente;

- $w_{j,i}^{(k)}$ : peso che connette il neurone $j$ del layer $k-1$ al neurone $i$ del layer $k$;

- $w_{0,i}^{(k)}$ : bias del neurone $i$ nel layer $k$

Ogni layer successivo non elabora più dati grezzi (x), ma le rappresentazioni astratte create dal layer precedente.

I primi layer potrebbero rilevare semplici bordi o gradienti.
I layer intermedi combinano i bordi in forme.
I layer finali riconoscono oggetti complessi.

Per capire questo concetto pensiamo a dei mattoncini lego:
- I dati grezzi ($X$) sono i singoli mattoncini sfusi sul pavimento;
- I **primi layer** uniscono i mattoncini per fare dei piccoli blocchi solidi (es. un muretto, una colonna);
- I **layer intermedi** uniscono le colonne e i muretti per costruire le stanze e le torri;
- I **layer finali** uniscono le stanze per dare la forma finale al castello.

La potenza di una Deep Neural Network, è quella di apprendere una gerarchia di caratteristiche attraverso queste trasformazioni nidificate. Più la rete è profonda, più è complessa la funzione che può approssimare.

# Il processo di apprendimento: Loss e Ottimizzazione
Addestrare una rete significa trovare i pesi W che minimizzano l'errore.
Ma come quantifichiamo l'errore?
Andiamo ad analizzare la funzione Loss:
$\mathcal{L}(\hat{y}, y)$

Questa funzione misura quanto $\hat{y}$ è lontana da $y$. Ma come è composta questa Loss function?

- Mean Squared Error (MSE): Usata per la regressione (es. predire il prezzo di una casa). Si calcola il quadrato della differenza tra predetto e reale;

- Binary Cross Entropy: Usata per la classificazione (es. "Sì" o "No"). Misura quanto la probabilità predetta si discosta dal target reale;

$$J(W) = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)}; W\right), y^{(i)}\right)$$

$J(W)$ misura la Loss media sulla nostra rete neurale, 1/n serve a normalizzarla, per ottenere il valore medio di Loss della rete. Senza 1/n sarebbe la Loss totale sulla nostra rete.

Quando applichiamo la **Mean Squared Error**, l'obiettivo della rete è predire un valore numerico continuo. Vogliamo quantificare di quanto il nostro risultato numerico si discosta da quello reale.

$$J(\mathbf{W}) = \frac{1}{n} \sum_{i=1}^{n} \left( y^{(i)} - f\left(x^{(i)}; \mathbf{W}\right)\right)^2$$

Immaginiamo di voler predire il voto esatto che uno studente prenderà all'esame (un valore da 18 a 30). Se la rete predice $24$ ($\hat{y}$) ma lo studente prende $28$ ($y$), la MSE misura la distanza lineare tra questi due numeri reali, la eleva al quadrato per penalizzare l'errore e dice alla rete come correggersi.

Quando applichiamo la **Binary Cross-Entropy**, l'obiettivo cambia completamente: non cerchiamo un numero qualunque, ma una probabilità strettamente compresa tra 0 e 1. Vogliamo misurare di quanto si discosta la certezza calcolata dalla rete rispetto alla realtà dei fatti.

$$J(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log\left(f\left(x^{(i)}; \mathbf{W}\right)\right) + \left(1 - y^{(i)}\right) \log\left(1 - f\left(x^{(i)}; \mathbf{W}\right)\right) \right]$$

Immagina di voler predire se lo studente supererà o meno l'esame (un problema binario: 1 = Passato, 0 = Bocciato). Se la rete sputa fuori $0.95$ ($\hat{y}$), sta dicendo che lo studente ha il 95% di probabilità di passare. Se lo studente viene inaspettatamente bocciato ($y = 0$), la BCE calcola quanto la rete è stata "punita" per aver dato una probabilità così alta a un evento che si è rivelato falso.

## Un focus sulla Binary Cross Entropy
A prima vista può sembrare intimidatoria, ma in realtà è molto semplice da comprendere. Smontiamola per capire esattamente come funziona.

Il target reale $y^{(i)}$ in una classificazione binaria può assumere solo due valori: **$1$** (evento vero) oppure **$0$** (evento falso).
All'interno della parentesi quadra notiamo due blocchi distinti separati da un segno $+$. Il valore di $y^{(i)}$ funge da vero e proprio **interruttore software**:

- **Se il target reale è $y^{(i)} = 1$ (Lo studente passa l'esame):**
  Il secondo blocco si annulla completamente perché contiene il fattore $(1 - y^{(i)})$, ovvero $(1 - 1) = 0$. 
  La formula si riduce a valutare solo il primo pezzo: **$1 \cdot \log(\hat{y})$**.
- **Se il target reale è $y^{(i)} = 0$ (Lo studente viene bocciato):**
  Il primo blocco si annulla perché moltiplicato direttamente per $y^{(i)} = 0$. Il secondo blocco invece si attiva poiché $(1 - 0) = 1$. 
  La formula si riduce a valutare solo il secondo pezzo: **$1 \cdot \log(1 - \hat{y})$**.

Grazie a questo trucco algebrico, un'unica riga di codice riesce a gestire contemporaneamente e in modo pulito sia i casi positivi che quelli negativi.

### Perché si usa il Logaritmo ($\log$)?

Il motivo fondamentale per cui introduciamo il logaritmo è legato al concetto di **distanza tra distribuzioni di probabilità** e alla forma geometrica che vogliamo dare alla nostra superficie d'errore. 

L'output della nostra rete, $\hat{y} = f(x^{(i)}; \mathbf{W})$, è una probabilità ed è quindi un numero strettamente confinato tra $0$ e $1$.
Andiamo a vedere cosa succede al logaritmo in questo intervallo:

* Il logaritmo di $1$ è pari a $0$: $\log(1) = 0$.
* Quando ci avviciniamo a $0$, il logaritmo tende a meno infinito: $\log(\rightarrow 0) = -\infty$.



Abbinando questa proprietà geometrica al **segno meno ($-$)** posto all'inizio della formula, otteniamo una punizione (una Loss) che si comporta in modo asimmetrico ed estremamente severo:

#### Caso A: Il target reale è $y = 1$
La rete deve tirare fuori un valore il più vicino possibile a $1$.
* Se la rete indovina e dice $\hat{y} = 0.99$, calcoliamo $-\log(0.99) \approx 0.01$ $\rightarrow$ **La Loss è quasi zero** (la rete viene premiata).
* Se la rete prende una sbandata incredibile e dice $\hat{y} = 0.01$ (sicura al 99% che l'evento sia falso), calcoliamo $-\log(0.01) \approx 4.60$. Man mano che la predizione si avvicina allo zero reale, il valore schizza verso l'infinito $\rightarrow$ **La Loss diventa gigantesca**.

#### Caso B: Il target reale è $y = 0$
La rete deve cacciare fuori un valore il più vicino possibile a $0$, il che significa che il termine $(1 - \hat{y})$ deve avvicinarsi a $1$.
* Se la rete indovina e dice $\hat{y} = 0.01$, il termine diventa $-\log(1 - 0.01) = -\log(0.99) \approx 0.01$ => **La Loss è minima**.
* Se la rete sbaglia e dice $\hat{y} = 0.99$ (sicura che lo studente passi, ma viene bocciato), calcoliamo $-\log(1 - 0.99) = -\log(0.01) \approx 4.60$ => **La Loss esplode di nuovo**.

### Vantaggi della BCE rispetto alla MSE nella classificazione

Perché non usare la semplice differenza al quadrato (MSE) anche per la classificazione?.
Ci sono due motivi cruciali per l'AI Engineering:

1. **Massima penalizzazione per l'eccesso di sicurezza errato:** Se una rete sbaglia una classificazione usando la MSE, l'errore massimo per un singolo campione è $(1 - 0)^2 = 1$. La rete non viene punita abbastanza se è "arrogante". La BCE, grazie alla curva logaritmica, tende a una punizione infinita. Questo costringe i pesi a cambiare istantaneamente.
2. **Stabilità matematica dei Gradienti:** Le funzioni di attivazione usate per la classificazione (come la *Sigmoide*) si appiattiscono alle estremità (quando l'output è vicino a 0 o 1), portando le derivate a zero. Se usassimo la MSE, l'addestramento si bloccherebbe perché i gradienti diventerebbero troppo piccoli (Vanishing Gradient). Il logaritmo della BCE cancella matematicamente questo appiattimento, mantenendo il flusso dei gradienti sempre vivo durante il processo di ottimizzazione.

Noi vogliamo trovare le network weights che riducono al minimo l'errore:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)}; \mathbf{W}\right), y^{(i)}\right)$$

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} J(\mathbf{W})$$

## La Backpropagation
Il termine backpropagation risponde alla domanda: Se cambio leggermente un peso $w_j$, di quanto cambierà l'errore finale $J(W)$?

Ricordiamoci che la nostra Loss è una funzione dei pesi della rete.

Immaginiamo di avere questa funzione, e prendiamo un punto a caso sulla nostra funzione dei pesi.

A questo punto calcoliamo il gradiente in questo punto scelto in maniera randomica. Una volta calcolato il gradiente (che ci dà la direzione in cui la nostra Loss aumenta, punta verso il massimo), noi ci muoviamo nella direzione opposta al gradiene, in maniera tale da muoverci verso il minimo, dove l'errore tende a essere più basso.

Continuiamo a ripetere questo processo finché non convergiamo al minimo.

### L'Algoritmo del gradient descent

- Assegniamo ai pesi dei valori casuali estratti da una distribuzione normale: $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$ (es. con deviazione standard $\sigma = 0.02$).
- Calcoliamo il vettore del gradiente della funzione di costo rispetto ai pesi: $\nabla_{\mathbf{W}} J(\mathbf{W})$, e aggiorniamo i pesi muovendoti nella direzione opposta al gradiente:
     $$\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} J(\mathbf{W})$$
- Restituiamo i pesi ottimali $\mathbf{W}^*$ quando l'algoritmo converge (ovvero quando la Loss smette di scendere).

Il parametro $\eta$ (eta) rappresenta il **Learning Rate** (tasso di apprendimento). Regola la grandezza del "passo" che facciamo lungo la discesa. Se è troppo grande rischiamo di scavalcare il minimo; se è troppo piccolo, l'addestramento impiegherà giorni. Anche per determinare il learning rate correttamente esiste un modo.

### La Chain Rule (Regola della Catena)

Tecnicamente, per aggiornare un singolo peso $w_j$, vogliamo calcolare la derivata parziale del costo rispetto a quel peso:

$$\frac{\partial J(\mathbf{W})}{\partial w_j}$$

In una rete neurale profonda (Deep Learning), dove l'input attraversa molti strati nidificati prima di generare l'output, calcolare questa derivata direttamente è impossibile. Per farlo, usiamo la **Chain Rule** (Regola della Catena) del calcolo infinitesimale.

Partiamo dall'errore generato sull'output finale e propaghiamo l'informazione all'indietro (*Backpropagation*), strato dopo strato. Ad esempio, la variazione dell'errore rispetto al peso del primo strato ($w_1$) si scompone nel prodotto delle variazioni locali:

$$\frac{\partial J(\mathbf{W})}{\partial w_1} = \frac{\partial J(\mathbf{W})}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$



### Spiegazione dei fattori:
1. $\frac{\partial J(\mathbf{W})}{\partial \hat{y}}$: Quanto varia l'errore globale al variare della predizione finale della rete.
2. $\frac{\partial \hat{y}}{\partial z_1}$: Quanto varia l'output del neurone rispetto al suo potenziale di attivazione interno ($z_1$).
3. $\frac{\partial z_1}{\partial w_1}$: Quanto varia il potenziale interno al variare del peso specifico $w_1$ (che dipende direttamente dall'input $x$).

Questo calcolo, ripetuto a ritroso per ogni singolo peso della rete, ci dice con precisione millimetrica in quale direzione "spostare" i miliardi di pesi del modello per scendere lungo la valle dell'errore e avvicinarci all'obiettivo di ottimizzazione.

### E' difficile ottimizzare la Loss?
Ottimizzare la Loss function può essere molto complesso.

Quello che abbiamo capito, è che l'ottimizzazione della Loss avviene attraverso la discesa del gradiente. Ma come facciamo in modo di scendere in maniera intelligente?

Dobbiamo stare attenti al Learning Rate, il termine $\eta$.
- Se scegliamo un Learning Rate troppo piccolo, convergeremo troppo lentamente al minimo, e resteremo bloccati probabilmente in un falso minimo (il learning rate è praticamente la grandezza del salto che vogliamo compiere sulla nostra funzione per capire se il punto in cui atterriamo può essere il minimo oppure no);
- Se lo scegliamo troppo alto, e quindi siamo predisposti ad un salto gigante, probabilmente finiremo per divergere più che convergere. Rischiamo di far esplodere il modello.
- Learning rate adattivi: invece di un LR fisso, facciamo in modo di variarlo basandosi sulla pendenza del terreno (della funzione) e sulla velocità con cui stiamo imparando.

#### Quali sono le idee?
1. Proviamo diversi learning rates e vediamo quali funzionano meglio;
2. utilizziamo il learning rate adattivo di cui abbiamo appena parlato.

Quindi il LR non sarà più fisso ma dipenderà da diversi fattori, come:
- quanto è largo il gradiente;
- quanto stiamo imparando velocemente;
- la size di alcuni pesi particolari;
- ...

Di seguito una serie di algoritmi di discesa del gradiente, presenti su TensorFlow e PyTorch:
- SGD
- Adam
- Adadelta
- Adagrad
- RMSProp

## Mini-Batches e SGD
Calcolare il gradiente su milioni di dati è lentissimo. Per questo usiamo lo Stochastic Gradient Descent (SGD) su piccoli gruppi di dati chiamati Mini-batches, un insieme di punti B:
- Velocità: calcoli più rapidi che sfruttano il parallelismo delle GPU
- Regolarizzazione: il "rumore" introdotto dal calcolare il gradiente solo su pochi esempi aiuta la rete a non bloccarsi in minimi locali poco profondi.

L'obiettivo dell'algoritmo è navigare la superficie matematica dell'errore per trovare la combinazione ottima di pesi $\mathbf{W}^*$ che minimizza il costo globale, aggiornando i parametri a piccoli passi (Batch).


#### Inizializzazione dei pesi
$$\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$$
L'algoritmo inizia assegnando ai pesi dei valori casuali estratti da una distribuzione normale con media 0 e una piccola deviazione standard $\sigma^2$ (ad esempio $\sigma = 0.02$). Questo serve a rompere la simmetria dei neuroni e permettere l'inizio dell'apprendimento.

#### Loop fino alla convergenza
Il processo entra in un ciclo iterativo che si ripete finché l'errore non si stabilizza vicino al minimo globale.

#### Campionamento del Mini Batch
Invece di elaborare tutti gli $n$ campioni del dataset, l'algoritmo estrae in modo casuale un piccolo blocchetto di dati composto da $B$ elementi (chiamato *Batch Size*, tipicamente 32, 64 o 128 campioni).

#### Calcolo del gradiente locale
$$\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}} = \frac{1}{B} \sum_{k=1}^{B} \frac{\partial J_k(\mathbf{W})}{\partial \mathbf{W}}$$
La rete calcola la derivata parziale della funzione di costo rispetto ai pesi ($\frac{\partial J}{\partial \mathbf{W}}$) facendo la media dei gradienti calcolati esclusivamente sui $B$ elementi del mini-batch corrente. 
* **Vantaggio:** Questo calcolo è estremamente veloce da eseguire sulla memoria della GPU/CPU rispetto al gradiente calcolato sull'intero dataset, garantendo un'apprendimento dinamico e reattivo.

#### Aggiornamento dei pesi
$$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$$
I pesi attuali vengono modificati sottraendo il gradiente calcolato nel passo precedente, moltiplicato per il **Learning Rate**.
Il segno meno ($-$) è fondamentale: poiché il gradiente punta verso la direzione di massimo aumento dell'errore come abbiamo già detto, quindi muoversi nella direzione opposta garantisce di scendere lungo la valle verso il punto di errore minimo.

#### Ritorno dei Pesi Ottimali
Quando l'algoritmo converge e la Loss smette di ridursi, il ciclo si interrompe e il modello restituisce la matrice finale dei pesi ottimizzati ($\mathbf{W}^*$).

Per capire l'esigenza ingegneristica dei mini-batch, immaginiamo uno studente che deve studiare un manuale d'esame pesantissimo di **1.000 pagine** (il nostro intero dataset). L'obiettivo dello studente è capire i concetti e correggere il proprio metodo di studio (aggiornare i pesi $\mathbf{W}$) per prendere 30 (minimizzare la Loss).

Ci sono tre modi in cui lo studente può approcciare lo studio:

#### L'approccio unico Batch
Lo studente decide di leggere **tutte e 1.000 le pagine di fila** senza mai fermarsi, senza farsi domande e senza fare quiz di autovalutazione. Solo dopo aver chiuso l'ultima pagina, si siede, fa un bilancio totale di cosa ha capito e corregge il suo metodo di studio.
* **Il problema:** È un processo lentissimo. Lo studente impiega settimane prima di fare un singolo aggiustamento. Inoltre, la sua memoria (la RAM del computer) viene sovraccaricata da una quantità enorme di informazioni concentrate in un colpo solo.

#### L'approccio Stocastico (SGD)
Lo studente legge **una singola riga della prima pagina** e si ferma immediatamente per fare un test. Se sbaglia il test su quella riga, stravolge completamente il suo metodo di studio. Poi legge la seconda riga, si ferma, rifà il test, cambia di nuovo metodo, e così via per 1.000 pagine.
* **Il problema:** Questo approccio è frenetico e caotico. Il metodo di studio cambia continuamente direzione a ogni singola riga leggendo dettagli insignificanti o eccezioni, creando un percorso spezzettato che oscilla continuamente senza mai trovare una strategia di studio equilibrata.

#### L'approccio Mini-Batch, ovvero la via di mezzo
Lo studente trova il compromesso perfetto: divide il libro in **capitoletti da 32 o 64 pagine** (la nostra *Batch Size*, $B$). 
Legge il primo blocchetto di pagine, si ferma, fa un quiz su quel capitolo e, in base al voto medio ottenuto, corregge e affina il suo metodo di studio. Poi passa al blocchetto successivo.

# Overfitting e Regolarizzazione
Uno dei problemi più comuni nell'addestramento di queste reti neurali è l'Overfitting: la rete impara a memoria i dati di addestramento (rumore incluso) ma fallisce su dati nuovi.

In realtà i problemi sono 2:
- Overfitting: di cui abbiamo appena parlato, la rete impara a memoria e non riesce a generalizzare, quindi a rispondere correttamente a domande che non ha imparato a memoria;
- Underfitting: quando il modello non ha la capacità di imparare bene dai dati, l'esempio che avevamo visto all'inizio del file che riguardava il bianco e nero. Il modello è troppo semplice, non riesce a cogliere la complessità dei dati e prende decisioni troppo nette e grossolane, esattamente come il perceptron senza bias che vede bianco e nero.

Bisogna quindi avere un fitting corretto, ovvero trovare il giusto compromesso tra semplicità e complessità.

Come si fa? Con la regolarizzazione.
La regolarizzazione è un insieme di tecniche che penalizzano la complessità del modello durante l'addestramento, impedendogli di memorizzare i dati di training e forzandolo a imparare pattern generalizzabili.

Vediamo alcune tecniche di regolarizzazione principali:
- Dropout;
- Early Stopping.
Possono essere anche usate in combinazione.

Il ***Dropout*** è una tecnica che consiste in: durante il training, spegniamo casualmente una percentuale di neuroni (es. il 50%) in ogni passaggio
Perché funziona? Costringe la rete a non fare affidamento su un singolo neurone specifico, distribuendo la conoscenza in modo più robusto su tutta l'architettura. Riduce l'overfitting proprio perché fa in modo che un neurone non si fidi ciecamente di quello che gli dicono gli altri, ma inizia a comprendere anche lui.

**Perché funziona?** Immaginiamo un team di sviluppo software aziendale composto da 5 persone in cui una sola persona (il "neurone genio") fa tutto il lavoro e gli altri 4 si limitano a guardare. Se il genio si ammala, il team fallisce (Overfitting su un unico elemento).
Il Dropout costringe l'azienda a mandare in ferie forzate e casuali i dipendenti ogni giorno. Se il genio oggi non c'è, gli altri 4 sono costretti a imparare a programmare e a collaborare tra loro. In questo modo, la conoscenza si distribuisce in modo robusto su tutta l'architettura: la rete impara a sopravvivere e a performare indipendentemente dalla presenza di un singolo neurone specifico.

***L'Early Stopping***: monitoriamo l'errore su un set di dati che la rete non ha mai visto (Testing set). Quando l'errore sul training scende ma quello sul testing inizia a risalire, fermiamo tutto: la rete sta iniziando a memorizzare invece di capire.

**Perché funziona?** Immagina uno studente dell'Unical che si esercita per settimane sugli stessi identici compiti degli esami degli anni passati. 
Nei primi giorni, lo studente capisce le regole generali della materia (la Loss scende sia sui compiti passati che su eventuali domande nuove poste dal professore).
Dopo la terza settimana di studio matto e disperatissimo, lo studente inizia a ricordarsi a memoria che *"Se la domanda ha la parola X, allora la risposta è la C"*. Ha smesso di ragionare. Se il professore all'esame cambia una singola virgola, lo studente fallisce.
L'Early Stopping agisce come un timer che toglie il libro dalle mani dello studente non appena si accorge che sta iniziando a imparare a memoria le risposte dei vecchi compiti anziché studiare la teoria.

**Abbiamo dunque visto:**
- il concetto di Perceptron (Percettone);
- il concetto di rete neurale: come passiamo dal percettone a una rete e l'ottimizzazione tramite back propagation;
- l'addestramento della rete neurale: addestramento adattivo, batching e regolarizzazione.
