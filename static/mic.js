/* Aveia — captação de microfone: constraints, medidor de nível, avisos de qualidade, gravação em pedaços
 * (IndexedDB), Wake Lock e recuperação de gravações interrompidas.
 *
 * Exposto como `window.Mic`. Depende só de DOM/Web APIs; a interface (index.html) liga as peças:
 *   Mic.abrir(cfg)              -> MediaStream (constraints, fallback de dispositivo, erros em pt-BR)
 *   Mic.monitorar(stream, ctx)  -> mede nível, mostra avisos, segura Wake Lock
 *   Mic.gravador(stream, nota)  -> MediaRecorder (mimeType/bitrate com fallback) que guarda pedaços no IndexedDB
 *   Mic.abrirSistema()          -> áudio do computador (getDisplayMedia), para gravar em canal separado do microfone
 *   Mic.parar()                 -> libera monitor e Wake Lock
 *   Mic.recuperar(guardarFn)    -> converte gravações interrompidas (aba fechada/crash) em "pendentes"
 *   Mic.iniciarAjustes(cfg, salvar) -> "Testar microfone" + perfil de processamento nos Ajustes
 */
(() => {
    'use strict';

    // ---------------------------------------------------------------- perfis de processamento do navegador
    // eco = echoCancellation, ruido = noiseSuppression, ganho = autoGainControl (ver README: medição de WER).
    const PERFIS = {
        voz:   { rotulo: 'Voz (recomendado)',            eco: true,  ruido: false, ganho: true  },
        bruto: { rotulo: 'Bruto (sem processamento)',    eco: false, ruido: false, ganho: false },
        total: { rotulo: 'Máximo (eco + ruído + ganho)', eco: true,  ruido: true,  ganho: true  },
    };
    const PERFIL_PADRAO = 'voz';
    const BITRATE = { opus: 64000, aac: 96000 };   // fala: 64 kbps Opus é transparente para ASR; AAC precisa de mais
    const BITRATE_ESTEREO = { opus: 128000, aac: 128000 };   // microfone + computador em canais separados (2 x ~64 kbps)
    const CANAIS_MIC_SISTEMA = 'mic+sistema';   // valor enviado ao servidor: esquerda = microfone, direita = computador

    const $ = id => document.getElementById(id);
    const dbfs = v => 20 * Math.log10(Math.max(v, 1e-6));
    let notificar = (msg, erro) => console[erro ? 'warn' : 'log'](msg);   // trocado por `aviso` da interface

    // ---------------------------------------------------------------- erros legíveis
    function explicarErro(e) {
        const n = e && e.name;
        if (!window.isSecureContext || !navigator.mediaDevices)
            return 'O navegador só libera o microfone em páginas seguras (localhost ou https). Abra o Aveia por http://127.0.0.1:8765 ou use HTTPS.';
        if (n === 'NotAllowedError' || n === 'SecurityError')
            return 'Acesso ao microfone negado. Clique no cadeado ao lado do endereço, permita o microfone e tente de novo (e confira as permissões de privacidade do sistema).';
        if (n === 'NotFoundError' || n === 'OverconstrainedError')
            return 'Nenhum microfone encontrado. Conecte um microfone ou fone e tente de novo.';
        if (n === 'NotReadableError' || n === 'AbortError')
            return 'Não consegui abrir o microfone: ele pode estar em uso por outro aplicativo (Teams, Zoom, Meet...) ou bloqueado pelo sistema.';
        return 'Microfone indisponível' + (e && e.message ? ` (${e.message})` : '') + '.';
    }

    // ---------------------------------------------------------------- abrir o microfone
    function restricoes(cfg, deviceId) {
        const p = PERFIS[cfg.micPerfil] || PERFIS[PERFIL_PADRAO];
        const audio = {
            channelCount: { ideal: 1 },
            echoCancellation: p.eco, noiseSuppression: p.ruido, autoGainControl: p.ganho,
        };
        // `exact`: o Chrome ignora `ideal` para deviceId (medido: abriu o padrão). Se o aparelho sumiu, o
        // OverconstrainedError é tratado em `abrir`, que cai para o padrão e avisa.
        if (deviceId) audio.deviceId = { exact: deviceId };
        return { audio };
    }

    async function pedir(cfg, deviceId) {
        try {
            return await navigator.mediaDevices.getUserMedia(restricoes(cfg, deviceId));
        } catch (e) {
            // O driver recusou as restrições extras (não o dispositivo): tenta só o dispositivo.
            if (e.name === 'OverconstrainedError' && e.constraint !== 'deviceId')
                return navigator.mediaDevices.getUserMedia({ audio: deviceId ? { deviceId: { exact: deviceId } } : true });
            throw e;
        }
    }

    async function abrir(cfg) {
        if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('sem mediaDevices'), { name: 'SecurityError' });
        if (!cfg.mic) return pedir(cfg, '');
        try {
            const s = await pedir(cfg, cfg.mic);
            const r = s.getAudioTracks()[0].label; if (r) cfg.micRotulo = r;
            return s;
        } catch (e) {
            if (e.name !== 'OverconstrainedError' && e.name !== 'NotFoundError') throw e;
        }
        // O id salvo não existe (fone desplugado; Safari/Firefox trocam ids): tenta achar o mesmo aparelho pelo rótulo.
        try {
            const igual = cfg.micRotulo && (await navigator.mediaDevices.enumerateDevices())
                .find(d => d.kind === 'audioinput' && d.label === cfg.micRotulo && d.deviceId !== cfg.mic);
            if (igual) { const s = await pedir(cfg, igual.deviceId); cfg.mic = igual.deviceId; return s; }
        } catch {}
        // Mantém a escolha salva (o fone pode voltar) e grava com o microfone padrão desta vez.
        const s = await pedir(cfg, '');
        notificar('O microfone escolhido não está disponível. Usando: ' + (s.getAudioTracks()[0].label || 'padrão do sistema'), true);
        return s;
    }

    // ---------------------------------------------------------------- áudio do computador (aba/tela compartilhada)
    /* Pede o compartilhamento de aba/tela COM áudio. Chamar no início do clique de "Gravar", antes de qualquer `await`:
     * o seletor do navegador exige a ativação do usuário, que expira.
     * Sempre resolve (nunca lança): { faixa, fluxo } quando há trilha de áudio; { motivo } caso contrário, com
     * motivo = 'sem-suporte' | 'cancelado' | 'sem-audio' | 'erro' (a gravação segue só com o microfone).
     *
     * A trilha de VÍDEO fica viva e sem uso (só reduzimos a taxa de quadros). Não existe garantia documentada (testada aqui)
     * de que parar o vídeo de um getDisplayMedia mantém o áudio no Chromium: aba/tela e áudio compartilham a mesma sessão de
     * captura e o botão "Parar compartilhamento" encerra as duas. O custo de deixar o vídeo vivo é pequeno (1 quadro/s). */
    async function abrirSistema() {
        if (!navigator.mediaDevices?.getDisplayMedia) return { motivo: 'sem-suporte' };
        const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };   // som do computador, sem tratamento
        const extras = { systemAudio: 'include', selfBrowserSurface: 'exclude', surfaceSwitching: 'include' };   // Chromium: oferece áudio do sistema, esconde a própria aba
        let fluxo;
        try {
            try { fluxo = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 1, max: 5 } }, audio, ...extras }); }
            catch (e) {
                if (e && e.name !== 'OverconstrainedError' && e.name !== 'TypeError') throw e;   // restrição não aceita: tenta o básico
                fluxo = await navigator.mediaDevices.getDisplayMedia({ video: true, audio });
            }
        } catch (e) {
            return { motivo: e && (e.name === 'NotAllowedError' || e.name === 'AbortError' || e.name === 'SecurityError') ? 'cancelado' : 'erro', erro: e };
        }
        const faixa = fluxo.getAudioTracks()[0];
        if (!faixa) { fluxo.getTracks().forEach(t => t.stop()); return { motivo: 'sem-audio' }; }
        return { faixa, fluxo };
    }

    // ---------------------------------------------------------------- medidor + avisos
    let monitor = null;
    function criarAlerta() {
        let a = $('mic-alerta');
        if (a) return a;
        a = document.createElement('div');
        a.id = 'mic-alerta';
        a.className = 'hidden fixed bottom-[4.75rem] inset-x-0 z-20 flex justify-center px-4 pointer-events-none';
        a.innerHTML = '<div class="pointer-events-auto max-w-md flex items-start gap-2 px-3.5 py-2 rounded-2xl bg-framboesa text-aveia text-xs font-medium shadow-lg">' +
            '<i class="ph-fill ph-warning text-base shrink-0 mt-px"></i><span id="mic-alerta-txt"></span></div>';
        document.body.appendChild(a);
        return a;
    }
    function mostrarAlerta(texto) {
        const a = criarAlerta();
        a.classList.toggle('hidden', !texto);
        if (texto) $('mic-alerta-txt').textContent = texto;
    }
    // Erro de permissão/dispositivo: texto longo, então fica 10 s (o toast da interface some em 3,5 s).
    let timerErro;
    function mostrarErro(texto) {
        mostrarAlerta(texto); clearTimeout(timerErro);
        timerErro = setTimeout(() => { if (!monitor) mostrarAlerta(''); }, 10000);
    }

    /* Analisa o sinal a cada 250 ms (setInterval: continua com a aba em segundo plano, ao contrário de rAF).
     * Avisos: sem sinal/mudo, volume baixo, saturação (clipping), banda estreita (headset Bluetooth em modo chamada). */
    function monitorar(stream, ctx, aoNivel, ativa = () => true) {
        pararMonitor();
        const faixa = stream.getAudioTracks()[0];
        const an = ctx.createAnalyser(); an.fftSize = 2048; an.smoothingTimeConstant = 0;
        const fonte = ctx.createMediaStreamSource(stream); fonte.connect(an);
        const onda = new Float32Array(an.fftSize), espectro = new Float32Array(an.frequencyBinCount);
        const m = {
            an, fonte, faixa, ctx, ativa, tempo: 0, inicio: performance.now(),
            nivel: { rms: -120, pico: -120 },
            // contadores em passos de 250 ms
            semSinal: 0, baixo: 0, ativos: 0, saturado: 0, estreito: 0, falaTotal: 0,
            avisos: new Set(), ultimaFala: 0,
            stats: { picoMax: 0, rmsFala: [], saturadas: 0, quadros: 0 },
        };
        m.id = setInterval(() => passo(m, onda, espectro, aoNivel), 250);
        const mudo = () => atualizar(m), voltou = () => atualizar(m);
        faixa.addEventListener('mute', mudo); faixa.addEventListener('unmute', voltou);
        // Fone desplugado/dispositivo removido: a interface encerra e guarda a gravação; aqui só explicamos.
        faixa.addEventListener('ended', () => notificar('Microfone desconectado. A gravação foi salva até aqui.', true));
        m.limpar = () => { faixa.removeEventListener('mute', mudo); faixa.removeEventListener('unmute', voltou); };
        ctx.addEventListener('statechange', () => { if (ctx.state === 'suspended' || ctx.state === 'interrupted') ctx.resume().catch(() => {}); });
        monitor = m;
        segurarTela();
        return m;
    }

    function passo(m, onda, espectro, aoNivel) {
        if (m.ctx.state === 'suspended') m.ctx.resume().catch(() => {});   // iOS/Safari suspende o AudioContext
        m.an.getFloatTimeDomainData(onda);
        if (!m.ativa()) { m.semSinal = m.baixo = m.saturado = m.estreito = 0; atualizar(m); return; }   // pausado: sem avisos
        let soma = 0, pico = 0, sat = 0;
        for (const v of onda) { const a = Math.abs(v); soma += v * v; if (a > pico) pico = a; if (a >= 0.95) sat++; }
        const rms = Math.sqrt(soma / onda.length);
        m.nivel = { rms: dbfs(rms), pico: dbfs(pico) };
        m.stats.quadros++; m.stats.picoMax = Math.max(m.stats.picoMax, pico);
        const ativo = m.nivel.rms > -50;                 // há fala/ruído audível neste quadro
        if (ativo) m.stats.rmsFala.push(m.nivel.rms);
        // sem sinal: tudo abaixo de -70 dBFS (microfone mudo no hardware, fone errado, driver sem entrada)
        m.semSinal = pico < 0.0003 ? m.semSinal + 1 : 0;
        // saturação: >= 0,3% das amostras acima de 0,95 (topo achatado) em quadros seguidos
        m.saturado = sat / onda.length > 0.003 ? m.saturado + 1 : Math.max(0, m.saturado - 1);
        // volume baixo: houve atividade, mas a fala nunca passa de -38 dBFS de pico
        if (ativo) { m.ativos++; m.baixo = m.nivel.pico < -38 ? m.baixo + 1 : 0; }
        // banda estreita: com fala forte, quase nada acima de 4 kHz (CVSD/SBC 8 kHz do perfil HFP)
        if (m.nivel.rms > -40 && m.tempo % 2 === 0) {
            m.an.getFloatFrequencyData(espectro);
            const hz = m.ctx.sampleRate / m.an.fftSize, soma = (a, b) => { let s = 0, n = 0; for (let i = Math.floor(a / hz); i <= Math.min(Math.floor(b / hz), espectro.length - 1); i++) { s += Math.pow(10, espectro[i] / 10); n++; } return s / Math.max(n, 1); };
            const baixa = soma(300, 3400), alta = soma(4500, Math.min(7500, m.ctx.sampleRate / 2 - 200));
            m.estreito = 10 * Math.log10(alta / Math.max(baixa, 1e-20)) < -50 ? m.estreito + 1 : Math.max(0, m.estreito - 1);
        }
        m.tempo++;
        if (aoNivel) aoNivel(m.nivel);
        if (m.tempo % 4 === 0) atualizar(m);
    }

    // Limiares em quadros de 250 ms; os avisos se desligam sozinhos quando a condição passa.
    function atualizar(m) {
        const av = m.avisos;
        const marca = (chave, cond) => cond ? av.add(chave) : av.delete(chave);
        marca('mudo', m.faixa.muted);
        marca('semsinal', m.semSinal >= 24 && !m.faixa.muted);          // 6 s sem sinal
        marca('baixo', m.baixo >= 20 && m.ativos >= 20);               // 5 s de fala fraca
        marca('saturado', m.saturado >= 4);                             // 1 s saturando
        marca('estreito', m.estreito >= 12);                            // 3 s de banda estreita
        const t = textoAviso(av);
        mostrarAlerta(t);
        if (m.aoAviso) m.aoAviso(t, av);
    }

    function textoAviso(av) {
        if (av.has('desconectado')) return 'Microfone desconectado. A gravação foi salva até aqui.';
        if (av.has('mudo')) return 'O microfone está mudo (botão do fone, chave do notebook ou configuração do sistema).';
        if (av.has('semsinal')) return 'Não estou recebendo som do microfone. Confira se ele está ligado e se é o dispositivo certo em Ajustes.';
        if (av.has('saturado')) return 'O som está estourando (saturação). Afaste-se do microfone ou reduza o volume de entrada.';
        if (av.has('baixo')) return 'Volume muito baixo. Aproxime-se do microfone ou aumente o volume de entrada no sistema.';
        if (av.has('estreito')) return 'O áudio chegou com banda estreita (comum em fone Bluetooth em modo chamada). Para melhor transcrição, use o microfone do notebook ou um fone com fio/USB.';
        return '';
    }

    function pararMonitor() {
        if (!monitor) return;
        clearInterval(monitor.id); monitor.limpar();
        try { monitor.fonte.disconnect(); } catch {}
        const r = resumoMonitor(monitor);
        monitor = null; mostrarAlerta('');
        liberarTela();
        return r;
    }

    function resumoMonitor(m) {
        const f = m.stats.rmsFala.slice().sort((a, b) => a - b);
        return { picoMax: dbfs(m.stats.picoMax), rmsFalaMediana: f.length ? f[Math.floor(f.length / 2)] : -120, avisos: [...m.avisos] };
    }

    // ---------------------------------------------------------------- Wake Lock (tela acesa durante a gravação)
    let trava = null;
    async function segurarTela() {
        try { if ('wakeLock' in navigator && !trava && document.visibilityState === 'visible') {
            trava = await navigator.wakeLock.request('screen');
            trava.addEventListener('release', () => { trava = null; });
        } } catch { trava = null; }
    }
    function liberarTela() { try { trava?.release(); } catch {} trava = null; }
    // O navegador solta o Wake Lock quando a aba fica oculta: pega de novo ao voltar.
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && monitor) segurarTela(); });

    // ---------------------------------------------------------------- gravação em pedaços (IndexedDB)
    // Banco separado do de pendentes ('aveia-audio'), para não mexer na migração dele.
    const Parciais = (() => {
        const abrir = () => new Promise((ok, no) => {
            const r = indexedDB.open('aveia-gravando', 1);
            r.onupgradeneeded = () => { r.result.createObjectStore('meta', { keyPath: 'id' }); r.result.createObjectStore('pedacos', { keyPath: ['id', 'seq'] }); };
            r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error);
        });
        const exec = async (lojas, modo, fn) => {
            const db = await abrir();
            return new Promise((ok, no) => {
                const t = db.transaction(lojas, modo); let res;
                try { res = fn(...lojas.map(l => t.objectStore(l))); } catch (e) { db.close(); return no(e); }
                t.oncomplete = () => { db.close(); ok(res && 'result' in res ? res.result : res); }; t.onerror = () => { db.close(); no(t.error); }; t.onabort = () => { db.close(); no(t.error); };
            });
        };
        return {
            meta: reg => exec(['meta'], 'readwrite', m => m.put(reg)),
            pedaco: (id, seq, blob) => exec(['pedacos'], 'readwrite', p => p.put({ id, seq, blob })),
            limpar: id => exec(['meta', 'pedacos'], 'readwrite', (m, p) => { m.delete(id); p.delete(IDBKeyRange.bound([id, 0], [id, Infinity])); }),
            metas: () => exec(['meta'], 'readonly', m => m.getAll()),
            blobs: async id => {
                const todos = await exec(['pedacos'], 'readonly', p => p.getAll(IDBKeyRange.bound([id, 0], [id, Infinity])));
                return todos.sort((a, b) => a.seq - b.seq).map(x => x.blob);
            },
        };
    })();

    const TIPOS = [
        ['audio/webm;codecs=opus', 'webm', 'opus'], ['audio/webm', 'webm', 'opus'],
        ['audio/mp4;codecs=mp4a.40.2', 'm4a', 'aac'], ['audio/mp4', 'm4a', 'aac'],
        ['audio/ogg;codecs=opus', 'ogg', 'opus'],
    ];
    function escolherTipo() {
        const t = TIPOS.find(([mime]) => window.MediaRecorder && MediaRecorder.isTypeSupported(mime));
        return t ? { mime: t[0], ext: t[1], codec: t[2] } : { mime: '', ext: 'webm', codec: 'opus' };
    }

    /* Cria o MediaRecorder e já liga a gravação em pedaços: a cada 2 s o pedaço vai para o IndexedDB.
     * Retorna { recorder, id, ext, finalizar(blob, duracao) } — `finalizar` deve ser chamado depois de a gravação
     * estar guardada como "pendente" (apaga os pedaços). */
    function gravador(stream, notaId, aoDados, canais) {
        const tipo = escolherTipo();
        const opts = { audioBitsPerSecond: (canais ? BITRATE_ESTEREO : BITRATE)[tipo.codec] };
        if (tipo.mime) opts.mimeType = tipo.mime;
        let recorder;
        try { recorder = new MediaRecorder(stream, opts); }
        catch { recorder = new MediaRecorder(stream); }
        const id = 'rec-' + Date.now();
        let seq = 0, falhou = false;
        const meta = { id, notaId, mime: recorder.mimeType || tipo.mime, ext: tipo.ext, inicio: Date.now(), vivo: Date.now(), duracao: 0 };
        if (canais) meta.canais = canais;   // layout do arquivo: sobrevive a um crash e vai ao servidor na recuperação
        Parciais.meta(meta).catch(() => { falhou = true; });
        recorder.addEventListener('dataavailable', e => {
            if (!e.data || !e.data.size) return;
            if (aoDados) aoDados(e.data);
            if (falhou) return;
            Parciais.pedaco(id, seq++, e.data)
                .then(() => Parciais.meta({ ...meta, vivo: Date.now(), duracao: (Date.now() - meta.inicio) / 1000 }))
                .catch(() => { falhou = true; });   // quota cheia: segue só em memória
        });
        return { recorder, id, ext: tipo.ext, tipo, finalizar: () => Parciais.limpar(id).catch(() => {}) };
    }

    // Gravações cuja aba foi fechada/travou: viram "pendentes" (aparecem em "Tentar de novo / Baixar áudio").
    async function recuperar(guardarPendente) {
        let n = 0;
        try {
            for (const m of await Parciais.metas()) {
                if (Date.now() - m.vivo < 30000) continue;          // ainda sendo gravada (outra aba)
                const blobs = await Parciais.blobs(m.id);
                if (blobs.length) {
                    const arquivo = new Blob(blobs, { type: m.mime || 'audio/webm' });
                    await guardarPendente({ id: m.id, arquivo, nomeArquivo: `gravacao.${m.ext}`, duracao: m.duracao || 0, notaId: m.notaId, em: m.inicio, ...(m.canais ? { canais: m.canais } : {}) });
                    n++;
                }
                await Parciais.limpar(m.id);
            }
        } catch (e) { console.warn('Recuperação de gravações interrompidas falhou', e); }
        return n;
    }

    // ---------------------------------------------------------------- Ajustes: testar microfone + perfil
    function iniciarAjustes(cfg, salvar, aoMudar) {
        const sel = $('cfg-mic');
        if (!sel || $('mic-extra')) return;
        const bloco = document.createElement('div');
        bloco.id = 'mic-extra'; bloco.className = 'space-y-2';
        bloco.innerHTML =
            '<label class="block"><span class="text-[11px] font-semibold uppercase tracking-wider text-amendoa">Processamento do navegador</span>' +
            '<select id="cfg-mic-perfil" class="mt-1 w-full rounded-xl bg-aveia/70 px-3 py-2 text-sm outline-none border border-transparent focus:border-amendoa/40"></select></label>' +
            '<div class="flex items-center gap-2"><button id="mic-testar" class="text-xs font-medium text-mirtilo/70 hover:text-mirtilo inline-flex items-center gap-1.5 shrink-0"><i class="ph ph-waveform"></i><span>Testar microfone</span></button>' +
            '<div class="flex-1 h-2 rounded-full bg-mirtilo/10 overflow-hidden"><div id="mic-medidor" class="h-full w-0 rounded-full bg-amendoa transition-[width] duration-100"></div></div></div>' +
            '<p id="mic-teste-txt" class="text-[11px] leading-snug text-mirtilo/50 hidden"></p>';
        sel.closest('label').after(bloco);
        const ps = $('cfg-mic-perfil');
        Object.entries(PERFIS).forEach(([k, p]) => ps.appendChild(new Option(p.rotulo, k)));
        ps.value = PERFIS[cfg.micPerfil] ? cfg.micPerfil : PERFIL_PADRAO;
        ps.addEventListener('change', () => { cfg.micPerfil = ps.value; salvar(); });
        $('mic-testar').addEventListener('click', () => testar(cfg));
        sel.addEventListener('change', () => { cfg.micRotulo = sel.selectedOptions[0]?.text || ''; salvar(); });
    }

    let teste = null;
    async function testar(cfg) {
        const txt = $('mic-teste-txt'), med = $('mic-medidor'), btn = $('mic-testar');
        if (teste) { teste.parar(); return; }
        txt.classList.remove('hidden');
        let s, ctx;
        try { s = await abrir(cfg); } catch (e) { txt.textContent = explicarErro(e); return; }
        ctx = new (window.AudioContext || window.webkitAudioContext)(); ctx.resume().catch(() => {});
        const emUso = monitor;                                // não derruba o monitor de uma gravação em andamento
        const an = ctx.createAnalyser(); an.fftSize = 1024;
        const fonte = ctx.createMediaStreamSource(s); fonte.connect(an);
        const onda = new Float32Array(an.fftSize);
        let pico = -120, t0 = performance.now(), picoMax = 0, sat = 0, n = 0;
        btn.querySelector('span').textContent = 'Parar teste';
        txt.textContent = 'Fale normalmente por alguns segundos: ' + (s.getAudioTracks()[0].label || 'microfone padrão');
        const id = setInterval(() => {
            an.getFloatTimeDomainData(onda);
            let p = 0, c = 0; for (const v of onda) { const a = Math.abs(v); if (a > p) p = a; if (a >= 0.99) c++; }
            const db = dbfs(p); pico = Math.max(pico, db); picoMax = Math.max(picoMax, p); if (c / onda.length > 0.001) sat++; n++;
            med.style.width = Math.max(0, Math.min(100, (db + 60) / 60 * 100)) + '%';
            med.className = 'h-full rounded-full transition-[width] duration-100 ' + (db > -3 ? 'bg-framboesa' : db < -45 ? 'bg-mirtilo/30' : 'bg-amendoa');
            if (performance.now() - t0 > 8000) teste.parar();
        }, 100);
        teste = { parar() {
            clearInterval(id); s.getTracks().forEach(t => t.stop()); try { fonte.disconnect(); } catch {} ctx.close().catch(() => {});
            med.style.width = '0'; btn.querySelector('span').textContent = 'Testar microfone';
            txt.textContent = veredito(pico, sat / Math.max(n, 1));
            teste = null;
        } };
    }
    function veredito(picoDb, fracSat) {
        if (picoDb < -65) return 'Nenhum som detectado. Confira se o microfone está ligado e é o dispositivo certo.';
        if (fracSat > 0.05) return `Pico ${picoDb.toFixed(0)} dBFS: o som está estourando. Afaste-se do microfone ou reduza o volume de entrada.`;
        if (picoDb < -30) return `Pico ${picoDb.toFixed(0)} dBFS: volume baixo. Aproxime-se ou aumente o volume de entrada no sistema.`;
        return `Pico ${picoDb.toFixed(0)} dBFS: nível bom.`;
    }

    // ---------------------------------------------------------------- lista de dispositivos
    /* Preenche o <select>. Os rótulos só existem depois de uma permissão: se vierem vazios, pede uma vez. */
    async function listar(sel, cfg) {
        try {
            if (!navigator.mediaDevices?.enumerateDevices) throw Object.assign(new Error(), { name: 'SecurityError' });
            let devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
            if (devs.length && !devs[0].label) {
                (await navigator.mediaDevices.getUserMedia({ audio: true })).getTracks().forEach(t => t.stop());
                devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
            }
            sel.replaceChildren(new Option(devs.length ? 'Padrão do sistema' : 'Nenhum microfone encontrado', ''));
            devs.filter(d => d.deviceId !== 'default' && d.deviceId !== 'communications')
                .forEach((d, i) => sel.appendChild(new Option(d.label || `Microfone ${i + 1}`, d.deviceId)));
            if (cfg.mic && ![...sel.options].some(o => o.value === cfg.mic)) {
                // id mudou (Safari/Firefox rotacionam ids): acha pelo rótulo. Se o aparelho sumiu, mantém a escolha salva.
                const igual = cfg.micRotulo && [...sel.options].find(o => o.text === cfg.micRotulo);
                if (igual) cfg.mic = igual.value;
            }
            sel.value = cfg.mic || '';
        } catch (e) {
            sel.replaceChildren(new Option(e.name === 'NotFoundError' ? 'Nenhum microfone encontrado' : 'Acesso ao microfone negado', ''));
        }
    }

    window.Mic = {
        PERFIS, abrir, abrirSistema, monitorar, gravador, recuperar, listar, explicarErro, iniciarAjustes, testar, escolherTipo,
        CANAIS_MIC_SISTEMA,
        parar: pararMonitor, mostrarErro,
        usarAviso: fn => { notificar = fn; },
        nivel: () => monitor ? monitor.nivel : null,
        avisos: () => monitor ? [...monitor.avisos] : [],
        _Parciais: Parciais,
    };
})();
