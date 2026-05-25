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

Per superare i limiti delle reti classiche e permettere all'input di dipendere dal passato e dallo stato precedente di sé stesso, nascono le Recurrent Neural Networks (RNNs).

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
