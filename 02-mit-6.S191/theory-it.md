# Deep Sequence Modelling
Come modelliamo i dati sequenziali, dove l'ordine e il contesto temporale sono requisiti fondamentali?

Quali sono degli esempi di sequenze in natura?
- i suoni;
- i video;
- il testo;
- il dna;
- il movimento;
- etc.

Le architetture per costruire delle reti neurali sequenziali sono 4:
- one to one: dato un input statico, associamo un output statico (dati i voti di uno studente, predire se passerà o no l'esame);
- many to one: data una sequenza, gli associamo un output statico (dato un tweet, associare ad esso un sentiment positivo o negativo);
- one to many: dato un input statico gli associamo una sequenza (data un'immagine, associargli un testo);
- many to many: data una sequenza gli associamo un'altra sequenza (dato un testo in italiano, tradurlo in inglese).

A questo punto, dobbiamo fare delle considerazioni sui vecchi modelli, che non possiamo più utilizzare quando entra in gioco il tempo.
Dobbiamo considerare infatti i singoli passi temporali (time steps).

Per superare i limiti delle reti classiche e permettere allo stato interno di dipendere dal passato e dallo stato precedente di sé stesso, nascono le Recurrent Neural Networks (RNNs).

Perché il perceptron e la rete Feed-forward (o dense layer, multi layer) non vanno più bene?
- Flusso unidirezionale(assenza di cicli): Nelle reti classiche l'informazione viaggia in una sola direzione. L'input entra a sinistra e l'informazione si sposta verso destra fino all'output, senza alcun ciclo all'indietro. Ad esempio: una rete che classifica immagini di radiografie: riceve i pixel, li elabora strato per strato, e produce "tumore o non tumore". L'informazione va solo in avanti, non c'è nessun motivo per cui guardare indietro, ogni immagine è indipendente.
Nelle RNN, invece, lo stato della rete viaggia sia in avanti verso l'output sia all'indietro ritornando su se stesso, in modo da essere rianalizzato e considerato al passo temporale successivo. Ad esempio: la frase "Il calcio è uno sport violento", "calcio" può essere il minerale, l'azione, o lo sport. Per capirlo dobbiamo leggere "sport" che viene dopo. Una rete classica non può tornare indietro a rivalutare "calcio" alla luce di ciò che ha letto dopo.

- Assunzione di Indipendenza(Mancanza di Memoria): Come diretta conseguenza del primo punto, la rete Feed-forward assume che l'input in una determinata posizione sia completamente indipendente rispetto a ciò che è avvenuto prima. Ogni time step azzera il cervello del modello. Ad esempio: predire il voto di uno studente dati i suoi parametri (ore di studio, presenze, voti precedenti): ogni studente è indipendente dagli altri, non c'è contesto temporale da ricordare.
Nelle sequenze reali il significato di un elemento dipende totalmente dal contesto precedente e dallo stato precedente di sé stesso(la memoria accumulata), che le reti classiche distruggono a ogni passo temporale. Ad esempio: generare una risposta a "Mi chiamo Marco, ho 25 anni. Che lavoro mi consigli?", senza memoria del contesto, il modello risponde senza ricordarsi che l'utente si chiama Marco e ha 25 anni. Ogni parola è scollegata dalla precedente.


# Recurrent Neural Networks (RNNs)
Una Rete Neurale Ricorrente (RNN) è una classe di reti neurali artificiali progettata specificamente per l'elaborazione di dati sequenziali.
A differenza delle reti Feed-Forward tradizionali, che elaborano l'intero input in un unico istante isolato, una RNN opera passo dopo passo nel tempo (time step), processando gli elementi della sequenza uno alla volta in modo cronologico.

Il termine **Ricorrente** deriva dalla presenza di un ciclo di retroazione (o feedback loop) all'interno della struttura della rete.

Invece di limitarsi a trasmettere l'informazione in avanti verso lo strato successivo, lo strato ricorrente prende il risultato del calcolo effettuato al tempo attuale e lo reimmette dentro se stesso come input per il calcolo del passo successivo.

Questo continuo "ritornare su se stessa" permette alla rete di sviluppare una memoria interna. Grazie a questo meccanismo, la risposta della rete a un determinato input non dipende esclusivamente dall'elemento che sta leggendo in quel preciso millisecondo, ma è influenzata dall'intera storia degli elementi passati che ha già incontrato lungo la sequenza.

Per mettere in pratica l'idea di far ritornare l'informazione su se stessa e creare una reale dipendenza dal passato, le RNN introducono una variabile di memoria interna chiamata Hidden State (Stato Nascosto), indicata con $h_t$.

A ogni singolo passo temporale $t$, la rete applica una funzione matematica $f_W$ che prende l'input attuale $x_t$ e lo unisce allo stato precedente di sé stesso $h_{t-1}$:

$$h_t = f_W(h_{t-1}, x_t)$$

Ovvero, abbiamo appena descritto matematicamente la memoria. Ma perché $$f_W$$?
Quella $W$ rappresenta i pesi della rete neurale che definiscono la funzione stessa, quindi è una funzione dei pesi. Non sono i pesi a cambiare, ma gli input ad ogni singolo passo temporale. I pesi (parametri) rimangono sempre uguali mentre la rete legge la sequenza passo passo, cambieranno solo quando andremo a fare la backpropagation.

Il momento in cui elaboriamo in maniera sequenziale è il Forward, quello della backpropagation è il Backward.

A ogni singolo istante $t$ (quando la rete legge una nuova parola, un nuovo frame di un video o un nuovo millisecondo di un audio), la rete prende la vecchia lavagna ($h_{t-1}$), ci aggiunge le informazioni del presente ($x_t$), cancella i dettagli inutili e scrive la nuova versione della lavagna ($h_t$).

Le RNN hanno uno stato, $h_t$, che viene aggiornato a ogni passo temporale mentre una sequenza viene elaborata.

## Come funziona la rete all'istante t?

1. La cella riceve due vettori dall'esterno:

    $x_t$ (Input Vector): Il dato del presente (es. i numeri che rappresentano la parola attuale).

    $h_{t-1}$ (Old Hidden State): Il vettore che arriva dal passato, contenente la memoria dei passi precedenti.

2. La cella prende questi due vettori e li dà in pasto alla formula di aggiornamento. 
Moltiplica l'input per i suoi pesi ($W_{xh} \cdot x_t$).
Moltiplica il passato per i suoi pesi ($W_{hh} \cdot h_{t-1}$).
Somma i risultati (e il bias) e li stringe dentro la tanh.
Il risultato di questo calcolo è il nuovo Hidden State ($h_t$). Questo vettore ha una doppia vita: viene mandato in avanti nel tempo, diventando il passato per il prossimo time step ($t+1$). Viene mandato verso l'alto per calcolare l'output del presente.
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

3. La rete non si limita a ricordare, deve anche produrre una risposta (Output Vector). 
Per farlo, prende il nuovo stato appena calcolato ($h_t$) e lo moltiplica per la terza matrice di pesi, chiamata $W_{hy}h_t$.
Se la rete deve indovinare la parola successiva, questo $y_t$ passerà per una funzione per sputare fuori la probabilità della parola corretta.
$$y_t = W_{hy} h_t + b_y$$

### Come funziona quindi la RNN completa essenzialmente?

Le tre matrici fondamentali non cambiano mai da un passo all'altro: $W_{xh}$ elabora l'input corrente ($x_0, x_1, x_2, \dots, x_t$), $W_{hh}$ trasmette la memoria da una cella alla successiva, $W_{hy}$ proietta lo stato interno verso l'output (${\hat{y}}_0, {\hat{y}}_1, {\hat{y}}_2, \dots, {\hat{y}}_t$)

Mentre la sequenza avanza da sinistra a destra: 

Al passo $t=0$, l'input $x_0$ genera la predizione ${\hat{y}}_0$. 

Questa predizione viene immediatamente confrontata con il target reale per calcolare la prima perdita parziale: $L_0$.

Al passo $t=1$, l'input $x_1$ si unisce alla memoria precedente tramite $W_{hh}$ e genera la seconda predizione ${\hat{y}}_1$, producendo la perdita: $L_1$.

Questo processo si ripete identico per ogni singolo istante fino alla fine della sequenza ($t$).

## Quali sono i criteri di progettazione per costruire una RNN efficace?

1. Gestire sequenze di lunghezza variabile (Variable-length sequences): 
In natura, le sequenze non hanno mai una dimensione fissa. 
Se un modello di classificazione di immagini riceve sempre foto di $224 \times 224$ pixel, un modello di linguaggio deve poter leggere sia una frase di 3 parole, sia un intero romanzo di 100.000 parole. 
L'architettura non deve avere vincoli rigidi sulla dimensione dell'input. 
Deve poter elaborare sequenze lunghe a piacere senza dover ridisegnare la struttura della rete o cambiare il numero di parametri.

2. Tracciare le dipendenze a lungo termine (Long-term dependencies):
Il significato di una parola alla fine di una pagina può dipendere totalmente da una parola scritta all'inizio del capitolo. 
Il modello deve avere una struttura di memoria capace di mantenere l'informazione intatta anche quando tra due elementi correlati ci sono decine o centinaia di passaggi intermedi.

3. Mantenere l'ordine dei dati: nelle sequenze, l'ordine cambia completamente il significato del messaggio. 
Le due frasi "Il cane morde il ragazzo" e "Il ragazzo morde il cane" contengono esattamente le stesse identiche parole, ma descrivono due situazioni opposte. 
La rete deve elaborare le informazioni rispettando rigorosamente la cronologia temporale, in modo che la rappresentazione finale rifletta chi compie l'azione e chi la subisce in questo caso. 

4. Condividere i parametri lungo la sequenza: ad esempio se la parola "calcio" si trova all'inizio della frase o alla fine, la regola fondamentale per elaborarla e capirne il significato non deve cambiare. Non possiamo addestrare parametri diversi per ogni singola posizione del testo, altrimenti il modello diventerebbe gigantesco e non saprebbe cosa fare se incontrasse una frase più lunga del previsto. Quindi i pesi e le regole di calcolo della rete devono essere condivisi (gli stessi) per ogni passo temporale. Questo rende il modello flessibile e gli permette di generalizzare le regole grammaticali o strutturali a prescindere da quando un elemento appare nella sequenza.

## Un problema sequenziale: predire la prossima parola in una frase.

"Questa mattina ho portato il mio cane a passeggio"

Data in input "Questa mattina ho portato il mio cane a", predici la prossima parola.

Una considerazione importante da fare, è che le reti neurali non prendono parole in input, ma soltanto numeri. Quindi dobbiamo trasformare le nostre parole in vettori di numeri di taglia fissa.

Tre fasi:

- **tokenizzazione**: dividiamo una frase in parole (in realtà l'intero vocabolario delle parole di un linguaggio pure):
input: "Questa mattina ho portato il mio cane a"
output: ["Questa", "mattina", "ho", "portato", "il", "mio", "cane", "a"];

- **indexing**: ad ogni parola associamo un numero:
Il modello ha un dizionario interno fisso, chiamato vocabolario, che contiene tutte le parole che ha imparato durante l'addestramento. In questa fase, ogni token viene cercato nel vocabolario e sostituito con il suo rispettivo numero intero ID (l'indice).
output: [1, 2, 3, 4, 5, 6, 7, 8];

- **one-hot embedding**: ad ogni numero associamo un vettore:
Questo strato sostituisce ogni ID intero con un vettore di zeri tranne
l'output per la parola "cane" ($x_7$): [0, 0, 0, 0, 0, 0, 1]

Al posto del one-hot embedding avremmo potuto usare il learned embedding, una sorta di mappa per capire quanto delle parole sono correlate tra loro: ad esempio le parole che hanno relazioni semantiche strette o correlate tra loro (es. "cane", "gatto", "cucciolo") avranno vettori con valori numerici molto simili, posizionandosi vicine nello spazio geometrico tracciato dalla rete.

Come abbiamo visto prima inoltre, c'è differenza nel senso della frase se non preservo l'ordine originale, era uno dei criteri di una RNN.

Quindi, i criteri per predire la prossima parola sono soddisfacibili da una RNN?
- Encoding Language for a Neural Network: Sì, come abbiamo appena visto;
- Handle Variable Sequence Lengths: sì, le RNNs sono progettate per accettare dimensioni dell'input variabili;
- Model Long-Term Dependencies: sì, come abbiamo visto prima;
- Capture Differences in Sequence Order: sì, mantenere l'ordine è uno dei criteri delle RNNs.

## Come addestriamo le RNN nella pratica?
Con la backpropagation, come per le semplici reti neurali. Ma è più complesso nel caso di reti sequenziali ricorrenti, e capiremo perché.

Come facevamo per le reti neurali non ricorrenti?
Dati degli input, passavamo dai pesi, producevamo degli output. A questo punto ci andavamo a calcolare la Loss (lo scostamento tra la predizione e il target reale). La backpropagation consisteva nell'andare a cacolare il gradiente di questa Loss function rispetto ai pesi della rete, spostarci nella direzione opposta del gradiente (gradient descent) e aggiornare i pesi per minimizzare la Loss.

Ora però il discorso è un po' più complesso, perché dobbiamo estendere questo concetto al contesto delle RNNs.
Come abbiamo detto, nel caso delle RNNs abbiamo questo processamento sequenziale step by step, quindi possiamo predire l'output ad ogni step temporale, e possiamo avere la Loss ad ogni time step.
Sommiamo tutta questa Loss e otteniamo una Loss unica.

Qui la backpropagation è diversa: invece di tornare indietro per ogni pezzo della rete ad ogni time step, quindi invece di fare la backpropagation classica ad ogni time step, facciamo una Backpropagation through time, ovvero:
non aggiorniamo nessun peso finché non abbiamo finito di leggere l'intera sequenza temporale, perché comunque l'errore ce lo portiamo indietro nel tempo. Prendiamo l'errore totale calcolato alla fine della frase ad esempio, e poi lo facciamo viaggiare all'indietro nella catena rispetto alla matrice dei pesi della memoria ($W_{hh}$), ovvero l'unica condivisa nel tempo.

Ritornando indietro nel tempo, sorge un problema, dobbiamo fare per ogni step delle moltiplicazioni di matrici, ovvero dei pesi della memoria, perché significherebbe derivare tutti gli stati intermedi a ritroso (per vedere come $h_0$ influenza la loss):

$$\frac{\partial h_3}{\partial h_0} = \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial h_0}$$

e poiché in ognuno di quei passaggi intermedi è nascosta la matrice dei pesi, ci ritroviamo a calcolare dei fattori accumulati di $W_{hh}$, come ad esempio $W_{hh}^3$

$$\frac{\partial h_3}{\partial h_0} \propto W_{hh} \times W_{hh} \times W_{hh} = W_{hh}^3$$

Cosa implica questo?

- Se i valori sono > 1 => exploding gradients (l'idea sarebbe quella di normalizzare i valori dei gradienti in un range, in maniera tale che il risultato abbia un massimo e non esploda, gradient clipping);
- Se i valori sono < 1 => vanishing gradients: ovvero il gradiente tenderebbe a zero e non riusciremmo a catturare dei problemi lontani dalla posizione finale, ma magari solo quelli vicini alla fine della rete, ad esempio con derivata $0.0000001 saremmo capaci di aggiornare solo i pesi delle parole vicine, la rete diventerebbe completamente cieca delle parole iniiali;

Ci sono diverse soluzioni per far fronte al vanishing gradient:

- funzioni di attivazione;
- inizializzazione dei pesi;
- architettura della rete.

Una soluzione a questo problema è stata quella del meccanismo di Gating nei neuroni. Ovvero, andiamo ad utilizzare dei gate per decidere se aggiugere o rimuovere informazioni dallo stato della memoria.

Il meccanismo di Gating risolve questo problema utilizzando delle porte logiche regolate dalla funzione di attivazione Sigmoide ($\sigma$). Poiché la sigmoide restituisce valori strettamente compresi tra 0 e 1, questi "gate" agiscono come dei rubinetti: se il gate restituisce 0: il canale è chiuso (l'informazione viene rimossa o bloccata). Se il gate restituisce 1: il canale è aperto (l'informazione passa liberamente e viene aggiunta o mantenuta).

# Le limitazioni delle RNNs per il sequence modelling.
- encoding bottleneck: ci portiamo dietro tutte queste informazioni del passato come stato interno della rete;
- troppo lente: non c'è parallelizzazione;
- memoria corta (vanishing).

Quindi le RNNs usano questo metodo step by step per tracciare la dipendenza dal tempo. 

Il nostro obiettivo sarebbe non avere queste limitazioni ed avere questi vantaggi:
- vedere la sequenza globalmente come uno stream continuo di informazioni (passare da "La parola 100 può parlare con la parola 1 solo attraverso i 98 passaggi intermedi della memoria" a "La parola 100 guarda direttamente la parola 1, saltando completamente tutti gli intermediari");
- non dovremmo andare a gestire la sequenza time step by time step, ma parallelizzarla;
- avere memoria lunga.

Come possiamo fare per ottenere questi vantaggi?
Un approccio potrebbe essere quello di comprimere tutto insieme, un unico vettore in input, prendere tutti i dati di tutti gli step temporali, concatenarli, ottenere il feature vector (ovvero il vettore embedded concatenato unico), produrre l'output.

Così eliminiamo la ricorrenza, ma non è scalabile per reti molto dense, molto grandi, perderemmo l'ordine inoltre, e come facciamo ad avere memoria senza concetto di tempo?

## Attension is all you need

I transformer sono la soluzione.
Ma cosa sono i transformers?

Una rete neurale può identificare quali parti dell'input sono più importanti? Quali rappresentano meglio l'input?

La soluzone è identificare quali sono le parti che hanno più rilevanza, e poi estrarre i contenuti più importanti che sono stati identificati come più rilevanti (ovvero con scores di attenzione alti).

Ma come facciamo a capire quali sono più rilevanti? Come una semplice ricerca classica.

Noi abbiamo una query, questa query produce dei risultati, cerchiamo nella chiave se c'è qualcosa di collegato alla nostra query.
Una volta che abbiamo identificato la parte importante dei nostri dati, andiamo ad estrarli.

L'obiettivo dell'**Attention Search** è calcolare la rilevanza reciproca tra tutte le parole della sequenza in un solo colpo (globalmente), permettendo alla rete di focalizzarsi sulle informazioni più importanti ed estrarle in modo mirato.

Il processo si articola in quattro fasi fondamentali:

### 1. Iniezione dell'Ordine (Positional Embedding)
Prima di effettuare qualsiasi calcolo di attenzione, dobbiamo risolvere il problema della mancanza di coordinate temporali dovuta alla parallelizzazione (visto che la rete legge tutta la frase contemporaneamente).

Prendiamo la matrice dei dati in input, dove ogni riga è il *learned embedding* semantico di una parola, e ci **sommiamo** il **Positional Embedding**:

Sono due vettori distinti di numeri, separati all'inizio, che poi vengono fusi insieme. Il primo vettore contiene i valori che rappresentano la semantica della parola, il secondo vettore contiene i valori che rappresentano la posizione della parola all'interno della frase.

$$\vec{x}_{\text{finale}} = \vec{x}_{\text{semantic}} + \vec{x}_{\text{position}}$$

ad esempio: 

$$
\vec{x}_{\text{semantic}} = [0.25, -0.87, 0.41, 0.12]
$$

$$
\vec{x}_{\text{position}} = [0.14, 0.95, -0.32, 0.88]
$$

Questo "timbro" posizionale viene applicato **una sola volta all'inizio**. Poiché le matrici Query ($Q$), Key ($K$) e Value ($V$) verranno generate successivamente partendo da questo unico input unificato, tutte e tre erediteranno automaticamente le informazioni sull'ordine delle parole.


### 2. Sdoppiamento dei Ruoli (Query, Key, Value)
La matrice di input (ora arricchita con le posizioni) viene moltiplicata per tre matrici di pesi differenti ($W_Q, W_K, W_V$) apprese durante l'addestramento.

A cosa servono questi pesi? All'inizio, la nostra parola ha un unico vettore ($\vec{x}_{\text{finale}}$) che fonde insieme significato e posizione. Tuttavia, una parola non può fare tutto insieme: non può proporsi come "soggetto", cercare un "oggetto" e contemporaneamente fornire il suo significato profondo usando gli stessi identici numeri.

Questa operazione proietta ogni parola in tre vettori con ruoli ben definiti che simulano un sistema di ricerca relazionale:

* **Query ($Q$) $\rightarrow$ *"Cosa sto cercando?"***: Rappresenta la parola attuale che interroga il resto della frase (la stringa digitata nella barra di ricerca).
* **Key ($K$) $\rightarrow$ *"Chi sono e cosa offro?"***: Rappresenta il tag di indicizzazione e le caratteristiche di ogni parola, usato per farsi trovare dalle altre (l'indice o il titolo nel database).
* **Value ($V$) $\rightarrow$ *"Il mio contenuto reale"***: Contiene l'informazione semantica profonda della parola, che verrà effettivamente estratta se c'è un buon match tra la Query e la sua Chiave.



### 3. Calcolo dei Pesi di Attenzione (The Match)
Per capire quali parti dell'input sono correlate tra loro, la rete calcola la somiglianza matematica tra ogni Query e tutte le Key disponibili sul tavolo tramite un prodotto scalare ($Q \cdot K^T$).

La somiglianza sarebbe la probabilità che questa parola sia correlata a quella della K, non che sia uguale. La somiglianza geometrica nel prodotto scalare indica la correlazione logica, grammaticale o semantica, ovvero quanto due parole siano necessarie l'una all'altra per completare il senso della frase.

1. I punteggi grezzi ottenuti vengono divisi per un fattore di scala ($\sqrt{d_k}$, dove $d_k$ è la dimensione dei vettori delle chiavi) per stabilizzare i calcoli ed evitare che i gradienti svaniscano durante la backpropagation.
2. I risultati scalati vengono dati in pasto a una funzione **Softmax**.

La Softmax trasforma i punteggi in una vera e propria classifica di percentuali (valori strettamente compresi tra $0$ e $1$, la cui somma fa sempre $1$). Questi sono gli **Attention Weights** (i pesi di attenzione), che indicano numericamente quanta rilevanza ha una determinata parola rispetto a un'altra nel contesto attuale.

Per capire come la matematica crei la mappa della rilevanza, mettiamoci nei panni del Transformer mentre analizza la frase:

**"Il programmatore scrive codice."**

Vogliamo calcolare l'attenzione rispetto alla parola **scrive** (il verbo),per capire quali altre parole della frase sono fondamentali per comprenderne il contesto.

Ipotizziamo che la dimensione dei vettori delle chiavi ($d_k$) scelta dai progettisti del modello sia **64**. Di conseguenza, il nostro fattore di scala sarà:

$$\sqrt{d_k} = \sqrt{64} = \mathbf{8}$$

Ecco i tre passaggi esatti che la GPU esegue in parallelo:

### Passo A: I Punteggi Grezzi (Il prodotto scalare $Q \cdot K^T$)
La Query della parola *scrive* si scontra tramite un prodotto scalare con le Key di tutte le parole della frase. Poiché i vettori sono ad altissima dimensione (64 elementi), le moltiplicazioni geometriche sommano molti numeri tra loro, generando dei punteggi grezzi alti e molto distanti tra loro:

* `scrive` $\times$ `Il` $\rightarrow$ **`8.0`** *(Basso: un articolo non dà contesto al verbo)*
* `scrive` $\times$ `programmatore` $\rightarrow$ **`56.0`** *(Altissimo! È il soggetto che compie l'azione)*
* `scrive` $\times$ `scrive` $\rightarrow$ **`24.0`** *(Medio: l'auto-riflessione della parola su se stessa)*
* `scrive` $\times$ `codice` $\rightarrow$ **`40.0`** *(Molto alto! È l'oggetto, ciò che viene scritto)*

### Passo B: La Scalatura
Se dessimo in pasto alla Softmax un numero enorme come `56.0`, la funzione esponenziale della Softmax ($e^{56}$) schizzerebbe a un valore quasi infinito, azzerando i gradienti di tutte le altre parole e mandando in tilt l'addestramento (*Vanishing Gradient*).

Per stabilizzare i calcoli, la rete divide tutti i punteggi per il fattore di scala, che in questo caso è **8**:

* **Il** $\rightarrow$ $8.0 / 8 = \mathbf{1.0}$
* **programmatore** $\rightarrow$ $56.0 / 8 = \mathbf{7.0}$
* **scrive** $\rightarrow$ $24.0 / 8 = \mathbf{3.0}$
* **codice** $\rightarrow$ $40.0 / 8 = \mathbf{5.0}$

I calcoli ora sono ridimensionati, protetti dalle deformazioni matematiche e pronti per il verdetto finale.

### Passo C: La Softmax 
La funzione Softmax prende i numeri scalati `[1.0, 7.0, 3.0, 5.0]`, ne calcola l'esponenziale e li normalizza, schiacciandoli in un intervallo tra $0$ e $1$ in modo che la loro somma totale faccia esattamente **1** (ovvero il **100%**).

I punteggi geometrici si trasformano ufficialmente nei veri **Attention Weights** (i pesi di attenzione):

| Parola (Key) | Calcolo Softmax | Peso di Attenzione (Rilevanza) | Cosa ha capito il Transformer |
| :--- | :--- | :--- | :--- |
| **Il** | $e^1 / \text{totale}$ | **0.2%** (`0.002`) | È un articolo, contesto trascurabile per il verbo. |
| **programmatore** | $e^7 / \text{totale}$ | **84.3%** (`0.843`) | **Soggetto.** È l'informazione cruciale per dare un senso a "scrive". |
| **scrive** | $e^3 / \text{totale}$ | **1.5%** (`0.015`) | Mantiene una minima quota di attenzione su se stesso. |
| **codice** | $e^5 / \text{totale}$ | **14.0%** (`0.140`) | **Oggetto.** È la seconda informazione fondamentale della frase. |
| **SOMMA** | | **100%** (`1.00`) | |


### 4. Estrazione del Contenuto (High Attention Extraction)
Una volta identificati i pesi, passiamo all'estrazione vera e propria descritta nell'intuizione classica della ricerca. Moltiplichiamo la matrice dei pesi di attenzione ottenuti dalla Softmax per la matrice dei **Value ($V$)**.

Le parole che hanno ottenuto un punteggio di attenzione alto (*high attention scores*) vedranno il proprio Value estratto quasi interamente e trasmesso in avanti agli strati successivi. Al contrario, le parti con punteggi vicini a zero verranno filtrate, attenuate e ignorate dal modello. 

### Passo D: L'estrazione Matematica
Abbiamo i nostri pesi di attenzione per la parola *"scrive"* calcolati al Passo C. Ora la GPU prende i vettori **Value ($V$)** di ogni parola (che contengono il patrimonio informativo profondo) e applica una media ponderata:

$$\vec{x}_{\text{output\_scrive}} = (0.002 \cdot \vec{V}_{\text{Il}}) + (0.843 \cdot \vec{V}_{\text{programmatore}}) + (0.015 \cdot \vec{V}_{\text{scrive}}) + (0.140 \cdot \vec{V}_{\text{codice}})$$

Cosa contiene, a livello di significato, questo nuovo vettore finale che la rete ha appena generato per la parola *"scrive"*?

* Il contenuto di **"Il"** è stato quasi del tutto filtrato e azzerato ($0.2\%$).
* Il contenuto originario di **"scrive"** conserva solo una piccolissima quota di auto-riflessione ($1.5\%$).
* Il vettore è ora letteralmente **inondato** dall'informazione semantica profonda di **"programmatore"** ($84.3\%$) e di **"codice"** ($14.0\%$).



Il risultato finale è un nuovo vettore di output in cui la parola iniziale non è più isolata, ma è stata **arricchita da tutto il contesto globale** della frase. Il verbo *"scrive"* ora non è più un'azione astratta: porta matematicamente dentro di sé l'informazione di *chi* sta compiendo l'azione e di *cosa* viene effettivamente scritto.

## La formula unificata dell'Attention Search

Tutti questi passaggi logici e matriciali si condensano matematicamente nell'equazione della **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Spacchettamento della formula:
* **$QK^T$** $\rightarrow$ **Fase 3 (Il Match):** Moltiplicazione geometrica di ogni Query contro tutte le Key per calcolare la somiglianza e la rilevanza grezza.
* **$\text{softmax}\left(\frac{\dots}{\sqrt{d_k}}\right)$** $\rightarrow$ **Fase 3 (I Pesi):** Scalatura dei valori e conversione dei punteggi in percentuali/probabilità stabili (da $0$ a $1$).
* **$\times V$** $\rightarrow$ **Fase 4 (L'Estrazione):** Estrazione selettiva e pesata dei contenuti più importanti basandosi sui punteggi alti della Softmax.

**Questo è il blocco principale per costruire l'architettursa dei Transformer.**

Questo approccio è incredibilmente parallelizzabile.

## Il Blocco di Calcolo: La "Self-Attention Head"

Tutte queste operazioni matematiche si collegano in modo sequenziale all'interno di un unico modulo hardware/software chiamato **Self-Attention Head**.

Graficamente, il flusso dei dati all'interno di una singola testa segue questa struttura:
1. L'input (arricchito dal *Positional Encoding*) entra nel blocco.
2. Viene sdoppiato attraverso tre strati lineari paralleli per generare le tre matrici **Query**, **Key** e **Value**.
3. Le matrici $Q$ e $K$ subiscono il prodotto scalare ($MatMul$) e vengono scalate ($\sqrt{d_k}$).
4. Viene applicata la **Softmax** per ottenere la matrice dei pesi.
5. Il risultato viene moltiplicato ($MatMul$) per la matrice $V$, sputando fuori il vettore finale arricchito dal contesto globale.

Questa intera struttura rappresenta il blocco fondamentale dell'architettura Transformer. 
Non ci sono cicli temporali, non c'è ricorrenza: l'intera frase viene elaborata in un colpo solo.

## Multi-Head Attention

Abbiamo un limite cruciale nella singola testa: una sola Attention Head può concentrarsi su una sola relazione alla volta. Ad esempio, nella frase "Ha lanciato la pallina da tennis per battere", la parola *"battere"* deve legarsi sia a *"tennis"* sia a *"lanciato"*.

Per permettere alla rete di guardare la frase da più punti di vista contemporaneamente, i Transformer applicano la **Multi-Head Attention**, ovvero fanno lavorare più teste di calcolo **in parallelo**.


* **Testa 1** si concentra sulle relazioni sintattiche e grammaticali (soggetto $\rightarrow$ Verbo).
* **Testa 2** si concentra sulle relazioni semantiche a lungo termine (contesto dello spazio).
* **Testa 3** analizza la struttura dei complementi o della punteggiatura.

## Campi di Applicazione (Self-Attention Applied)

Questo meccanismo di calcolo spaziale parallelo si è rivelato così potente da uscire dai confini del testo, rivoluzionando qualsiasi dato che possa essere espresso come una sequenza:

* **Language Processing (NLP):** È la base di modelli come BERT, GPT e di tutti i moderni Large Language Models (LLM).
* **Computer Vision:** Nei *Vision Transformers (ViT)*, le immagini vengono divise in piccoli quadratini (*patch*) trattati esattamente come se fossero parole di una frase.
* **Sequenze Biologiche:** Modelli come *AlphaFold* sfruttano la Self-Attention per mappare le interazioni tra amminoacidi e predire il ripiegamento tridimensionale delle proteine e del DNA.
