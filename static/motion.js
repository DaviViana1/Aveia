/* =====================================================================
   Aveia — movimento (motion.js). Depende de static/motion.css (tokens de tempo e curvas).

   Princípios: nada aqui muda regras de negócio. O app é o dono do DOM; este arquivo só
   (1) envolve algumas funções de renderização para animar o antes/depois (FLIP, entradas),
   (2) observa mutações pontuais (cronômetro, botões, título) e
   (3) delega eventos (ripple, checkbox). Se uma função envolvida não existir mais, é ignorada.
   Só `transform`, `opacity` (e `height` em um elemento isolado, via WAAPI) são animados.
   API pública para testes e ajustes: window.Motion.
   ===================================================================== */
(() => {
    'use strict';
    const root = document.documentElement;
    try { if (localStorage.getItem('aveia.motion') === 'off' || /[?&]motion=off\b/.test(location.search)) root.classList.add('mo-off'); } catch {}
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    const reduced = () => mq.matches || root.classList.contains('mo-off');
    const $ = id => document.getElementById(id);
    const raf2 = fn => requestAnimationFrame(() => requestAnimationFrame(fn));
    raf2(() => root.classList.add('mo-ready'));   // libera as transições só depois do primeiro quadro

    // ---------- Tokens (lidos do CSS: um único lugar para ajustar) ----------
    const probe = Object.assign(document.createElement('div'), {});
    probe.style.cssText = 'position:fixed;left:-9px;top:0;width:1px;height:1px;visibility:hidden;pointer-events:none';
    document.body.append(probe);
    const ms = name => { probe.style.transitionDuration = `var(${name})`; const v = getComputedStyle(probe).transitionDuration; return parseFloat(v) * (/ms$/.test(v) ? 1 : 1000) || 0; };
    const D = { fast: () => ms('--mo-fast'), base: () => ms('--mo-base'), slow: () => ms('--mo-slow'), reveal: () => ms('--mo-reveal') };
    const ease = name => { probe.style.transitionTimingFunction = `var(--mo-ease-${name})`; return getComputedStyle(probe).transitionTimingFunction || 'ease'; };
    const E = n => ease(n);

    // anim(): Web Animations com fallback para reduced-motion (só fades curtos; sem deslocamentos)
    function anim(el, frames, opt = {}) {
        if (!el || !el.animate) return null;
        let { duration = D.base(), easing = E('out'), delay = 0, fill = 'backwards' } = opt;
        if (reduced()) {
            frames = frames.filter(f => 'opacity' in f).map(f => ({ opacity: f.opacity, ...(f.offset != null ? { offset: f.offset } : {}) }));
            if (frames.length < 2) return null;
            duration = Math.min(duration, 120); delay = 0;
        }
        try { return el.animate(frames, { duration, easing, delay, fill }); } catch { return null; }
    }
    const rise = (el, i = 0, dy = 8, dur) => anim(el, [{ opacity: 0, transform: `translateY(${dy}px)` }, { opacity: 1, transform: 'none' }], { duration: dur ?? D.slow(), delay: i * 45 });

    // Interpolação numérica com easing (contadores). Em reduced-motion salta para o valor final.
    function tween(dur, fn, alive = () => true) {
        if (reduced() || dur <= 0) return fn(1);
        const t0 = performance.now();
        (function f(now) { const p = Math.min(1, (now - t0) / dur); fn(1 - Math.pow(1 - p, 3)); if (p < 1 && alive()) requestAnimationFrame(f); })(t0);
    }

    // Envolve uma função global (declaração `function`), preservando o retorno.
    function wrap(name, before, after) {
        const orig = window[name];
        if (typeof orig !== 'function' || orig.__mo) return;
        const w = function (...a) {
            let pre; try { pre = before ? before(...a) : undefined; } catch (e) { console.warn('[motion]', name, e); }
            const r = orig.apply(this, a);
            try { after && after(pre, r, ...a); } catch (e) { console.warn('[motion]', name, e); }
            return r;
        };
        w.__mo = true; window[name] = w;
    }

    // Altura suave de um bloco cujo conteúdo mudou de uma vez (WAAPI no elemento isolado).
    function smoothHeight(el, from, to) {
        if (reduced() || !el.animate || Math.abs(to - from) < 3 || el.getAnimations().length) return;
        el.style.overflow = 'hidden';
        const a = el.animate([{ height: from + 'px' }, { height: to + 'px' }], { duration: D.base() + 60, easing: E('out') });
        const fim = () => { el.style.overflow = ''; };
        a.onfinish = fim; a.oncancel = fim;
    }

    let lastSwitch = 0;   // instante da última troca de nota (outras animações cedem a vez)

    // ---------- Ripple (toque/clique) ----------
    document.addEventListener('pointerdown', e => {
        if (reduced() || e.button) return;
        const b = e.target.closest && e.target.closest('button, [role="button"]');
        if (!b || b.disabled || b.classList.contains('mo-x') || b.id === 'rec-btn' || !b.animate) return;
        const r = b.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top;
        const rad = Math.hypot(Math.max(x, r.width - x), Math.max(y, r.height - y));
        const s = document.createElement('span');
        s.className = 'mo-rip';
        s.style.cssText = `left:${x}px;top:${y}px;width:${rad * 2}px;height:${rad * 2}px`;
        if (getComputedStyle(b).position === 'static') b.style.position = 'relative';
        b.classList.add('mo-rip-host');
        b.append(s);
        const a = s.animate([{ scale: 0, opacity: .26 }, { scale: 1, opacity: 0 }], { duration: D.slow() * 1.6, easing: E('out') });
        a.onfinish = a.oncancel = () => s.remove();
    }, true);

    // ---------- Menu lateral: FLIP, entradas, saídas (fantasmas), caret e pílula do item ativo ----------
    const menu = { pill: null, abertos: new Map(), pilha: false };
    const menuFechado = () => { const m = $('menu'); return !m || m.classList.contains('pointer-events-none'); };

    function menuSnap() {
        const nav = $('menu-lista'); if (!nav) return null;
        const nr = nav.getBoundingClientRect(), st = nav.scrollTop, map = new Map();
        nav.querySelectorAll('[data-k]:not(.mo-ghost)').forEach(e => {
            const r = e.getBoundingClientRect();
            map.set(e.dataset.k, { el: e, x: r.left - nr.left, y: r.top - nr.top + st, w: r.width, h: r.height });
        });
        let pill = null;
        if (menu.pill && menu.pill.isConnected && menu.pill.classList.contains('on')) {
            const r = menu.pill.getBoundingClientRect(); pill = { x: r.left - nr.left, y: r.top - nr.top + st };
        }
        return { map, pill, grupos: new Map(menu.abertos), fechado: menuFechado() };
    }

    function ghost(o) {
        const nav = $('menu-lista'), g = o.el;
        g.classList.add('mo-ghost'); g.removeAttribute('draggable'); g.setAttribute('aria-hidden', 'true');
        g.style.cssText += `;position:absolute;left:${o.x}px;top:${o.y}px;width:${o.w}px;height:${o.h}px;margin:0`;
        nav.append(g);
        const a = anim(g, [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateX(-10px) scale(.96)' }], { duration: D.fast() + 20, easing: E('in'), fill: 'forwards' });
        if (a) a.onfinish = a.oncancel = () => g.remove(); else g.remove();
    }

    function menuDepois(pre) {
        const nav = $('menu-lista'); if (!nav) return;
        const nr = nav.getBoundingClientRect(), st = nav.scrollTop, cur = new Map();
        nav.querySelectorAll('[data-k]:not(.mo-ghost)').forEach(e => {
            const r = e.getBoundingClientRect();
            cur.set(e.dataset.k, { el: e, x: r.left - nr.left, y: r.top - nr.top + st, w: r.width, h: r.height });
        });
        const animar = pre && pre.map.size && !pre.fechado && !menuFechado() && cur.size < 220;

        if (animar) {
            let n = 0;
            cur.forEach((c, k) => {
                const o = pre.map.get(k);
                if (o) {
                    const dy = o.y - c.y;
                    if (Math.abs(dy) > .5) anim(c.el, [{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: D.slow(), easing: E('out') });
                } else {
                    anim(c.el, [{ opacity: 0, transform: 'translateY(-6px) scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: D.base(), delay: Math.min(n++, 8) * 22 });
                }
            });
            pre.map.forEach((o, k) => { if (!cur.has(k)) ghost(o); });
        }

        // Caret dos grupos: gira entre fechado (0°) e aberto (90°)
        if (typeof state !== 'undefined' && Array.isArray(state.groups)) {
            state.groups.forEach(g => {
                const antes = pre && pre.grupos.get(g.id);
                if (animar && antes !== undefined && antes !== !!g.aberto) {
                    const c = cur.get('g' + g.id), i = c && c.el.querySelector('.ph-caret-right');
                    if (i) anim(i, [{ transform: `rotate(${antes ? 90 : 0}deg)` }, { transform: `rotate(${g.aberto ? 90 : 0}deg)` }], { duration: D.base() + 40, easing: E('pop') });
                }
                menu.abertos.set(g.id, !!g.aberto);
            });
        }

        // Pílula do item ativo (replaceChildren a remove a cada render: reinsere)
        if (!menu.pill) { menu.pill = document.createElement('div'); menu.pill.id = 'mo-pill'; menu.pill.setAttribute('aria-hidden', 'true'); }
        const pill = menu.pill;
        nav.prepend(pill); nav.classList.add('mo-pill-on');
        const act = typeof state !== 'undefined' && cur.get('n' + state.atual);
        if (!act) { pill.classList.remove('on'); return; }
        pill.style.width = act.w + 'px'; pill.style.height = act.h + 'px';
        const to = `translate(${act.x}px, ${act.y}px)`;
        pill.style.transform = to; pill.classList.add('on');
        if (animar && pre.pill) anim(pill, [{ transform: `translate(${pre.pill.x}px, ${pre.pill.y}px)` }, { transform: to }], { duration: D.slow(), easing: E('out') });
        else if (animar) anim(pill, [{ opacity: 0 }, { opacity: 1 }], { duration: D.base() });
    }

    // Fantasma de arrasto: um "cartão" no lugar do instantâneo padrão do navegador
    document.addEventListener('dragstart', e => {
        const t = e.target.closest && e.target.closest('#menu-lista [draggable="true"]');
        if (!t || !e.dataTransfer || !e.dataTransfer.setDragImage) return;
        try {
            const r = t.getBoundingClientRect(), g = document.createElement('div');
            g.className = 'mo-drag-ghost';
            [...t.children].forEach(c => { if (c.tagName !== 'BUTTON') g.append(c.cloneNode(true)); });
            g.style.width = Math.min(r.width, 240) + 'px';
            document.body.append(g);
            e.dataTransfer.setDragImage(g, Math.min(Math.max(e.clientX - r.left, 8), 232), g.offsetHeight / 2);
            setTimeout(() => g.remove(), 0);
        } catch {}
    }, true);

    // ---------- Troca de nota: entrada curta e em cascata (não bloqueia nada) ----------
    let shownId = null, passe = false;
    function entrarNota() {
        lastSwitch = performance.now();
        const alvos = [document.querySelector('#layout article'), $('resumo'), $('transcript')]
            .filter(e => e && !e.classList.contains('hidden') && !e.getAnimations().length);
        alvos.forEach((e, i) => rise(e, i, 10, D.slow()));
    }

    // ---------- Wrappers de renderização ----------
    function instalar() {
        shownId = typeof state !== 'undefined' ? state.atual : null;
        if (typeof state !== 'undefined' && Array.isArray(state.groups)) state.groups.forEach(g => menu.abertos.set(g.id, !!g.aberto));

        wrap('renderMenu', menuSnap, menuDepois);

        wrap('renderizar',
            () => { const sw = shownId !== null && shownId !== state.atual; passe = sw; return { sw }; },
            pre => { passe = false; if (pre && pre.sw) entrarNota(); shownId = state.atual; });

        // Resumo: skeleton sincronizado, progresso suave, revelação em cascata, altura do cartão
        const prevResumo = {}, prog = {};
        wrap('renderResumo', () => { const c = $('resumo'); return { h: c.classList.contains('hidden') ? 0 : c.offsetHeight }; }, pre => {
            const n = nota(), corpo = $('resumo-corpo'), card = $('resumo');
            if (!n || !corpo) return;
            const sk = corpo.querySelectorAll('.animate-pulse');
            const modo = sk.length ? 'gerando' : (n.resumo ? 'pronto' : 'vazio');
            const antes = prevResumo[n.id]; prevResumo[n.id] = modo;
            const d = -(performance.now() % 1600) + 'ms';
            sk.forEach(s => s.style.setProperty('--mo-d', d));

            if (modo === 'gerando') {
                const barra = corpo.querySelector('.overflow-hidden > div'), txt = corpo.querySelector('p.text-xs');
                if (barra) {
                    const para = Math.min(1, (parseFloat(barra.style.width) || 0) / 100), de = prog[n.id] ?? 0;
                    barra.style.width = '100%'; barra.style.transformOrigin = 'left center'; barra.style.transform = `scaleX(${para})`;
                    anim(barra, [{ transform: `scaleX(${de})` }, { transform: `scaleX(${para})` }], { duration: D.reveal() * 1.4, easing: E('out') });
                    const m = txt && /^(.*?)(\d+)%$/.exec(txt.textContent);
                    if (m) { const a = Math.round(de * 100), b = +m[2]; tween(D.reveal() * 1.4, p => { txt.textContent = m[1] + Math.round(a + (b - a) * p) + '%'; }, () => txt.isConnected); }
                    prog[n.id] = para;
                }
            } else delete prog[n.id];

            if (modo === 'pronto' && antes === 'gerando') {
                let i = 0;
                [...corpo.children].forEach(c => {
                    const lista = c.querySelector(':scope > .space-y-2');
                    if (lista) {
                        const cab = c.firstElementChild; cab.classList.add('mo-in'); cab.style.setProperty('--i', i++);
                        [...lista.children].forEach(it => { it.classList.add('mo-in'); it.style.setProperty('--i', Math.min(i++, 14)); });
                    } else { c.classList.add('mo-in'); c.style.setProperty('--i', Math.min(i++, 14)); }
                });
                const st = card.querySelector('.ph-sparkle');
                if (st) anim(st, [{ transform: 'scale(.5) rotate(-40deg)' }, { transform: 'scale(1.3) rotate(10deg)', offset: .6 }, { transform: 'none' }], { duration: D.reveal() + 120, easing: E('out') });
            }
            if (pre && pre.h && antes && antes !== modo && !card.classList.contains('hidden')) smoothHeight(card, pre.h, card.offsetHeight);
        });

        // Síntese: barras crescem, números contam (só na troca de nota ou quando os dados mudam)
        const sigs = {};
        wrap('renderSintese', null, () => {
            const n = nota(); if (!n) return;
            const sig = n.id + '|' + n.falas.length + '|' + (n.duracao || 0);
            const mudou = sigs[n.id] !== sig; sigs[n.id] = sig;
            if (!(passe || mudou)) return;
            [...$('synth-chips').children].forEach((c, i) => anim(c, [{ opacity: 0, transform: 'scale(.85)' }, { opacity: 1, transform: 'none' }], { duration: D.slow(), delay: 80 + i * 55, easing: E('pop') }));
            [...$('synth-bars').children].forEach((row, i) => {
                const b = row.querySelector('.overflow-hidden > div'), pc = row.lastElementChild;
                const delay = 140 + i * 70, dur = D.reveal() + 220;
                if (b) {
                    const alvo = parseFloat(b.style.width) / 100 || 0;
                    b.style.width = '100%'; b.style.transformOrigin = 'left center'; b.style.transform = `scaleX(${alvo})`;
                    anim(b, [{ transform: 'scaleX(0)' }, { transform: `scaleX(${alvo})` }], { duration: dur, delay, easing: E('out') });
                }
                const m = pc && /^(\d+)%$/.exec(pc.textContent);
                if (m && !reduced()) {
                    const alvo = +m[1]; pc.textContent = '0%';
                    setTimeout(() => tween(dur, p => { pc.textContent = Math.round(alvo * p) + '%'; }, () => pc.isConnected), delay);
                }
            });
        });

        wrap('renderAvatares', null, () => {
            if (!passe) return;
            [...$('pill-avatars').children].forEach((a, i) => anim(a, [{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'none' }], { duration: D.slow(), delay: 120 + i * 50, easing: E('pop') }));
        });

        wrap('renderTranscricao', () => ({ oculto: $('transcript').classList.contains('hidden') }), pre => {
            if (!(passe || (pre && pre.oculto))) return;
            const rows = [...$('turns').children].slice(0, 9);
            rows.forEach((r, i) => anim(r, [{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: D.slow(), delay: 90 + i * 30 }));
        });

        // Toast: o texto desliza ao trocar; sucesso "desenha" o ícone
        wrap('aviso', null, (_, __, msg, erro) => {
            const icon = $('toast-icon'), txt = $('toast-text'); if (!icon || !txt) return;
            icon.parentNode.querySelectorAll('.mo-ck').forEach(s => s.remove());
            if (erro) { icon.style.display = ''; anim(icon, [{ transform: 'scale(.5)' }, { transform: 'scale(1.25)', offset: .6 }, { transform: 'none' }], { duration: D.slow(), easing: E('out') }); }
            else {
                icon.style.display = 'none';
                const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('class', 'mo-ck'); s.setAttribute('aria-hidden', 'true');
                s.innerHTML = '<circle class="mo-ck-c" cx="12" cy="12" r="9.5"/><path class="mo-ck-t" d="M7.5 12.5l3 3 6-6.5"/>';
                icon.before(s);
            }
            anim(txt, [{ opacity: 0, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], { duration: D.base() });
        });

        // Capítulo → fala: destaque que pisca
        wrap('irParaFala', null, (_, __, seg) => {
            const linhas = [...$('turns').children];
            const alvo = linhas.find(r => +r.dataset.inicio >= seg - .5) || linhas[linhas.length - 1];
            if (!alvo) return;
            anim(alvo, [
                { backgroundColor: 'rgba(177,138,101,.42)', boxShadow: '0 0 0 8px rgba(177,138,101,.42)' },
                { backgroundColor: 'rgba(177,138,101,0)', boxShadow: '0 0 0 8px rgba(177,138,101,0)' }
            ], { duration: 1500, easing: E('out'), delay: 250 });
        });
    }

    // ---------- Observadores ----------
    function observadores() {
        // (O título automático já tem a própria transição, `.titulo-novo`, definida no index.html.)

        // Editor: edição ↔ renderizado sem salto de altura
        const body = $('note-body'), vista = $('note-view'), ed = $('note-edit');
        if (body && vista && ed) {
            let h = body.offsetHeight, quando = 0;
            new ResizeObserver(() => { if (!body.getAnimations().length) h = body.offsetHeight; }).observe(body);
            new MutationObserver(() => {
                if (quando === performance.now() || performance.now() - lastSwitch < 150) return;
                quando = performance.now();
                const novo = body.offsetHeight; smoothHeight(body, h, novo);
            }).observe(vista, { attributes: true, attributeFilter: ['class'] });
        }

        // Gravação: estado visual, ícones que giram, anel reagindo ao nível, dígitos que rolam
        const btn = $('rec-btn'), ri = $('rec-icon'), pi = $('pause-icon'), label = $('rec-label');
        if (btn && ri && pi && label) {
            let ultimoRi = ri.className, ultimoPi = pi.className, raf = 0, nivel = 0;
            const estado = () => ri.className.includes('ph-stop') ? (pi.className.includes('ph-play') ? 'pausado' : 'gravando') : ri.className.includes('spin') ? 'processando' : 'ocioso';
            const loop = () => {
                const bs = document.querySelectorAll('.wave-bar'); let s = 0;
                bs.forEach(b => { s += parseFloat(b.style.height) || 4; });
                const alvo = bs.length ? Math.max(0, Math.min(1, (s / bs.length - 4) / 14)) : 0;
                nivel += (alvo - nivel) * .35;
                btn.style.setProperty('--mo-lvl', nivel.toFixed(3));
                raf = requestAnimationFrame(loop);
            };
            const sync = () => {
                const st = estado(); if (btn.dataset.mo !== st) btn.dataset.mo = st;
                const gravando = st === 'gravando' && !reduced();
                if (gravando && !raf) raf = requestAnimationFrame(loop);
                if (!gravando && raf) { cancelAnimationFrame(raf); raf = 0; nivel = 0; btn.style.removeProperty('--mo-lvl'); }
                if (ri.className !== ultimoRi) { ultimoRi = ri.className; anim(ri, [{ transform: 'scale(.3) rotate(-90deg)', opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: D.base() + 100, easing: E('pop') }); }
                if (pi.className !== ultimoPi) { ultimoPi = pi.className; anim(pi, [{ transform: 'scale(.3) rotate(90deg)', opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: D.base() + 100, easing: E('pop') }); }
            };
            const ob = new MutationObserver(sync);
            [ri, pi].forEach(e => ob.observe(e, { attributes: true, attributeFilter: ['class'] }));
            sync();

            // Dígitos do cronômetro / porcentagem: só o caractere que muda "rola"
            let ant = label.textContent;
            const re = /^[\d:]+%?$/;
            const lo = new MutationObserver(() => {
                const t = label.textContent;
                if (t === ant) return;
                const velho = ant; ant = t;
                if (reduced()) return;
                if (re.test(t) && re.test(velho) && t.length === velho.length) {
                    const caixa = document.createElement('span'); caixa.className = 'mo-dgs';
                    const novos = [];
                    [...t].forEach((ch, i) => { const s = document.createElement('span'); s.className = 'mo-dg'; s.textContent = ch; caixa.append(s); if (ch !== velho[i]) novos.push(s); });
                    label.replaceChildren(caixa); lo.takeRecords();
                    novos.forEach(s => anim(s, [{ opacity: 0, transform: 'translateY(55%)' }, { opacity: 1, transform: 'none' }], { duration: D.base(), easing: E('out') }));
                } else anim(label, [{ opacity: 0, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }], { duration: D.base() });
            });
            lo.observe(label, { childList: true, characterData: true, subtree: true });
        }

        // Botão Gerar/Gerando/Refazer: largura e conteúdo trocam suavemente
        const magic = $('magic-btn');
        if (magic) {
            let w = magic.offsetWidth;
            new ResizeObserver(() => { if (!magic.getAnimations().length) w = magic.offsetWidth; }).observe(magic);
            new MutationObserver(recs => {
                if (recs.every(r => [...r.addedNodes, ...r.removedNodes].every(n => n.classList && n.classList.contains('mo-rip')))) return;
                const novo = magic.offsetWidth;
                if (!reduced() && Math.abs(novo - w) > 2 && !magic.getAnimations().length) {
                    magic.style.overflow = 'hidden'; magic.style.whiteSpace = 'nowrap';
                    const a = magic.animate([{ width: w + 'px' }, { width: novo + 'px' }], { duration: D.base(), easing: E('out') });
                    a.onfinish = a.oncancel = () => { magic.style.overflow = ''; magic.style.whiteSpace = ''; w = magic.offsetWidth; };
                }
                [...magic.children].filter(c => !c.classList.contains('mo-rip')).forEach(c => anim(c, [{ opacity: 0, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], { duration: D.base() }));
            }).observe(magic, { childList: true });
        }

        // Avisos de recuperação: só o que é novo entra (o app reconstrói a lista inteira)
        const pend = $('pendentes');
        if (pend) {
            const vistos = new Set();
            const marcar = () => [...pend.children].forEach((r, i) => {
                const sig = r.textContent;
                if (vistos.has(sig)) r.style.animation = 'none'; else { vistos.add(sig); r.style.animationDelay = i * 60 + 'ms'; }
            });
            new MutationObserver(marcar).observe(pend, { childList: true });
            marcar();
        }
    }

    // Marcar tarefa: caixa "pula" e o traço é desenhado (o DOM é reconstruído pelo app)
    document.addEventListener('click', e => {
        // O app reconstrói o DOM no próprio clique: a caixa clicada já saiu da árvore, então usa o caminho do evento.
        const cb = e.target;
        if (!cb || cb.type !== 'checkbox' || !cb.dataset || !cb.dataset.linha || !e.composedPath().includes($('note-view'))) return;
        const novo = document.querySelector(`#note-view input[data-linha="${cb.dataset.linha}"]`);
        if (!novo) return;
        novo.classList.add('mo-just');
        const txt = novo.nextElementSibling;
        if (txt) anim(txt, [{ opacity: novo.checked ? 1 : .5 }, { opacity: novo.checked ? .5 : 1 }], { duration: D.base() + 60 });
    });

    // Rolagem suave respeita reduced-motion também nas chamadas explícitas do app
    const sIV = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function (a) {
        if (reduced() && a && typeof a === 'object' && a.behavior === 'smooth') a = { ...a, behavior: 'auto' };
        return sIV.call(this, a);
    };

    instalar();
    observadores();

    window.Motion = {
        reduced, anim, D, E,
        speed(x) { root.style.setProperty('--mo-t', String(x)); },          // 1 = padrão, .5 = 2x mais rápido
        off() { try { localStorage.setItem('aveia.motion', 'off'); } catch {} root.classList.add('mo-off'); },
        on() { try { localStorage.removeItem('aveia.motion'); } catch {} root.classList.remove('mo-off'); },
    };
})();
