# Deep Generative Modeling
In questa sezione vedremo le basi della generative modeling. 

Vedremo come creare delle reti neurali, che non imparano soltanto pattern dai dati, ma che riescono a capire come modellare le distribuzioni dei dati, in modo tale da replicare queste distribuzioni per creare nuove istanze di dati.

Per vedere quanto sono potenti questi algorimti, vedremo un esempio di generazione ai, provate a indovinare quali di queste immagini non sono generate da AI:

![Quiz](../../assets/deep-learning-intro-course/04/quiz.png)

In questo caso, l'immagine reale è la C, la A e la B sono state utilizzando Nano Banana 2 di Gemini.

## Addestramento supervisionato vs Addestramento non supervisionato
Fino ad ora, fino a questo capitolo, abbiamo sempre trattato reti neurali in cui l'addestramento era supervisionato (Feed Forward, RNN, CNN), ovvero dato un input, era dato anche l'output reale, la label.
L'obiettivo è sempre stato imparare un mapping funzionale tra x (input) e y (label, output): per classificare, regression, object detection, semantic segmentation etc.

Ma nel Deep Learning c'è una classe di problemi molto più grande, ovvero l'addestramento non supervisionato. Dove noi diamo alla nostra rete solo i dati in input senza alcuna label, senza alcun output reale. E l'obiettivo è quello è di capire, trovare, un modello che catturi queste distribuzioni di dati, in modo da imparare la struttura principale di questi dati, ovvero la distribuzione stessa.
Ad esempio: clustering, dimensionality reduction.

Quindi come funziona il **generative modeling**?
Prende in input esempi da una distribuzione e impara un modello di rete neurale che rappresenta questa distribuzione di probabilità.

Questo è il concetto principale ed essenziale della generative AI: catturare e imparare distribuzioni di probabilità dai dati.

Possiamo utilizzare questo concetto in un paio di modi:
- stima di densità: dati dei dati che seguono una distribuzione: impariamo un modello di questa distribuzione, ovvero descriviamo il modello probabilistico secondo cui questi dati sono distribuiti;
- se abbiamo effettivamente imparato una distribuzione di probabilità, a questo punto, possiamo generare usando questa distribuzione, questo modello.

Ma come facciamo ad imparare queste distribuzioni a partire dai dati?

Impariamo le feature più importanti di un dataset e le usiamo per costruire una rappresentazione più accurata e robusta della distribuzione reale. 
Questo ci permette anche di fare quello che è chiamato debiasing.

Sappiamo infatti che in molti casi reali, i dataset raccolti contengono bias, ovvero distorsioni sistematiche nei dati, spesso dovute al modo in cui sono stati raccolti. 

Ad esempio, un dataset di volti usato per il riconoscimento facciale potrebbe contenere molte più immagini di persone di una certa etnia rispetto ad altre, oppure un dataset medico potrebbe essere composto prevalentemente da pazienti maschi. 

![Bias](../../assets/deep-learning-intro-course/04/bias.png)

I modelli addestrati su questi dataset finiscono per catturare e replicare gli stessi bias, producendo risultati meno accurati o addirittura discriminatori per i gruppi sottorappresentati.

Usando tecniche di generative modeling, possiamo fare in modo che il modello impari la distribuzione sottostante in modo più equilibrato, generando dati sintetici che correggono o bilanciano le distorsioni presenti nel dataset originale.

Un altro esempio imporante è quello della detection degli **Outlier**.
Ricordiamoci nuovamente, i modelli generativi vanno a catturare queste distribuzioni di probabilità dai dati. Nei casi di guida autonoma ad esempio, o pilotaggio automatico di un aereo, o nella medicina, possono esistere dei casi limite, dei casi che capitano pochissime volte (es. un animale che spunta sulla strada all'improvviso, nebbia fitta, etc), come facciamo a catturare queste casistiche? Come usiamo i modelli generativi per catturare questi eventi che potrebbero essere problematici, e correggere il comportamento?

### Perché i modelli generativi?
Il cuore dei modelli generativi è proprio questo: una volta appresa la distribuzione di probabilità dei dati, possiamo campionare da essa, ovvero estrarre nuove istanze che seguono quella stessa distribuzione, come se fossero dati reali.
Questo meccanismo è alla base di tutta la Generative AI moderna, e si applica a domini molto diversi:

- **linguaggio naturale**: i modelli come ChatGPT generano testo campionando da distribuzioni apprese su miliardi di parole. Il risultato sono risposte coerenti, traduzioni, riassunti, codice, etc;

- **immagini e video**;

- **molecole**: in ambito farmaceutico e biologico, i modelli generativi vengono usati per progettare nuove strutture molecolari, accelerando enormemente la ricerca su farmaci e proteine;

In tutti questi casi il principio è lo stesso: il modello non memorizza i dati, ma ne cattura la struttura profonda, e da quella struttura sa generare qualcosa di nuovo.

## Modelli a variabili latenti
Esistono principalmente due famiglie classiche di modelli generativi:

- **VAE** (Variational Autoencoders): erano il paradigma dominante fino a qualche anno fa, e restano fondamentali per capire i principi alla base della generative AI;

- **GAN** (Generative Adversarial Networks): hanno dominato la generazione di immagini per diversi anni, producendo risultati visivamente molto convincenti.

Recentemente una terza famiglia ha preso il sopravvento: i Diffusion Models, su cui si basano oggi sistemi come Stable Diffusion e DALL-E.

La domanda più importante però è: 

**cosa intendiamo per "caratteristiche più rilevanti" dei dati, e cosa significa latente?**

### Il mito della caverna
Per rispondere a questa domanda, usiamo una delle metafore filosofiche più famose: il mito della caverna di Platone.

![Mito](../../assets/deep-learning-intro-course/04/myth.jpg)

Immaginiamo dei prigionieri incatenati in una caverna, con le spalle rivolte il muro. 
Dietro di loro brucia un fuoco, e tra il fuoco e i prigionieri passano delle figure. 
I prigionieri vedono solo le ombre proiettate sulla parete davanti a loro, non vedono mai gli oggetti reali.
Per i prigionieri, quelle ombre sono la realtà. 
Ma noi sappiamo che le ombre sono solo una rappresentazione ridotta di qualcosa di più ricco e complesso che esiste altrove.

Nel generative modeling funziona in modo analogo: i dati che osserviamo (immagini, testi, suoni), sono come le ombre sulla parete. 
Sono la manifestazione visibile di strutture più profonde e astratte che non osserviamo direttamente. 
Queste strutture nascoste sono le variabili latenti.
Lo spazio latente è quindi la stanza dietro la caverna: un mondo compresso e astratto dove vivono le caratteristiche essenziali dei dati. 
Il modello generativo impara a muoversi in questo spazio, e campionando da esso, proietta nuove ombre sulla parete, ovvero genera nuovi dati.