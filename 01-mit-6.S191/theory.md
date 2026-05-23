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
I bordi intermedi combinano i bordi in forme.
I layer finali riconoscono oggetti complessi.

Per capire questo concetto pensiamo a dei mattoncini lego:
- I dati grezzi ($X$) sono i singoli mattoncini sfusi sul pavimento;
- I **primi layer** uniscono i mattoncini per fare dei piccoli blocchi solidi (es. un muretto, una colonna);
- I **layer intermedi** uniscono le colonne e i muretti per costruire le stanze e le torri;
- I **layer finali** uniscono le stanze per dare la forma finale al castello.

La potenza di una Deep Neural Network, è quella di apprendere una gerarchia di caratteristiche attraverso queste trasformazioni nidificate. Più la rete è profonda, più è complessa la funzione che può approssimare.
