# Le Reti Neurali Convoluzionali.
In questo capitolo ci concentreremo su una parte dell'intelligenza estremamente affascinante e che molti di noi danno per scontata: **la vista**.

E' il senso che più diamo per scontato ma che ci permette di fare gran parte delle nostre attività quotidiane: riconoscere oggetti e persone.

Cos'è la vista? E' banalmente, la capacità di riconoscere qualcosa, e capire dove sta, semplicemente guardando.
Ma nella realtà è molto di più che riconoscere qualcosa, riconoscere un'immagine. Ci permette di capire le dinamiche, ad esempio le dinamiche dell'ambiente in cui siamo immersi, di capire una serie di **informazioni**.

Il deep learning si occupa di capire tutta questa quantità spropositata di informazioni, anche analizzare le immagini rientra in questa quantità spropositata di informazioni.

Questa materia nello specifico si chiama computer vision. Dove viene applicata attualmente?
Nella robotica, nel mobile computing, nella medicina, accessibilità, guida autonoma.

Il deep learning ha sempre avuto, fin dall'inizio, una moltitudine di tipi di riconoscimento e approcci diversi per la computer vision. 

Focalizzandosi principalmente sulla classificazione e il riconoscimento: come ad esempio nel caso del rilevamento e del riconoscimento facciale e del riconoscimento di patologie nel campo medico.

## Cosa vedono i computer?
Come vengono rappresentate da un computer queste informazioni che noi raccogliamo con la vista?
I computer non hanno gli occhi, non vedono i colori, vedono solo numeri.

Quindi, cosa sono le immagini per un computer? Per un computer le immagini sono "solo" numeri. Sono matrici, matrici 2D nel caso di una foto in bianco e nero ad esempio. Ogni valore della matrice corrisponde alla luminosità della cella associata.
Questa però è una rappresentazione molto semplificata, ma comunque molto potente.

Se noi vogliamo coprire lo spazio dei colori, dobbiamo estendere lo spazio a 3 matrici 2D. Abbiamo tre colori primari: RGB (rosso, verde, blue), e ognuno di questi colori è rappresentato da una matrice 2D.

Quindi essenzialmente, le immagini in bianco e nero sono rappresentate da una singola matrice, quelle a colori da una matrice tridimensionale.

Abbiamo due tipi di algoritmi per quanto riguarda la computer vision:
- regressione: gli output sono valori continui; 
- classificazione: gli output sono una probabilità.

Un esempio di problema di classificazione: data un'immagine, capire se si tratta di una persona, di un paesaggio, di un oggetto etc, non facciamo altro che assegnare una probabilità alle diverse opzioni (numero fissato di opzioni).

Può un algoritmo di machine learning differenziare tra queste 3 classi di opzioni ad esempio? Come fa?
Per farlo, per identificare queste immagini, la nostra pipeline deve essere capace di capire cosa è unico, cosa differenzia ognuna delle nostre opzioni dalle altre.

Ma come facciamo ad individuare cosa è rilevante in un'immagine? Dobbiamo individuare prima le caratteristiche che sono rilevanti, quali sono queste caratteristiche che differenziano una nostra categoria dalle altre. 

Ad esempio, prendiamo 3 casistiche:
- caso di una persona: naso, occhi, bocca;
- caso di una macchina: ruote, luci, specchi, targa;
- caso di un'abitazione: porte, finestre, scale.

Quindi noi andiamo a individuare queste caratteristiche di basso livello, e poi le mettiamo insieme per poi classificare i gruppi ad alto livello.

Quindi come abbiamo detto, un modo per risolvere questo problema, è sfruttare le conoscenze di un particolare campo, ad esempio come abbiamo detto, estrarre le caratteristiche di basso livello e poi metterle insieme per riconoscere ad esempio l'oggetto ad alto livello.

Ma c'è un problema con questo approccio: gli umani danno troppe cose per scontato, è difficile per noi definire in maniera precisa le cose, ci sono molte variazioni: anche definire una casa è difficile: variano le dimensioni, la luce, etc.

Il problema a questo punto diventa: come facciamo a riconoscere queste caratteristiche di basso livello (quelle essenziali, quelle base) di una particolare categoria?

Quindi vogliamo un modo per estrarre queste caratteristiche base, e contemporaneamente capire quando una di queste estratte è una buona caratteristica.

Per risolvere questo problema possiamo usare un approccio basato su reti neurali: impariamo direttamente dai dati, dalla profondità della nostra rete e dei dati.

## Apprendimento delle caratteristiche visive
Le reti neurali ci consentono di imparare caratteristiche visive: dai dati di queste immagini capiamo quali sono i pattern principali di queste immagini.

Utilizzeremo le FCNN (Reti Neurali completamente connesse)

Il nostro input della rete è ad una dimensione, è un'immagine 2D, quindi un vettore di pixel