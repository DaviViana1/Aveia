/* Nomes dos locutores: troca de nome nos textos derivados (resumo, título) e escolha do locutor pelo horário.
   Funções puras, sem DOM: o navegador as usa pelo objeto global `Nomes` e os testes as rodam no Node (testes/test_nomes_js.py). */
(function (raiz) {
    'use strict';

    const LETRA = '\\p{L}\\p{N}_';
    const escapar = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Troca cada nome de `antigos` por `novo`, só como palavra inteira ("Locutor 1" não pega "Locutor 10"; "Carlos" não pega "Carlota").
    // Sem diferença de maiúsculas. Uma passada só: se o nome novo contém o antigo, não é trocado duas vezes.
    function trocar(texto, antigos, novo) {
        novo = String(novo || '').trim();
        if (typeof texto !== 'string' || !texto || !novo) return texto;
        const alvos = [...new Set((antigos || []).map(a => String(a || '').trim()).filter(a => a && a !== novo))].sort((a, b) => b.length - a.length);
        if (!alvos.length) return texto;
        const re = new RegExp(`(?<![${LETRA}])(?:${alvos.map(a => escapar(a).replace(/\s+/g, '\\s+')).join('|')})(?![${LETRA}])`, 'giu');
        return texto.replace(re, () => novo);
    }

    // Aplica `trocar` em todo o texto do resumo (parágrafos, pontos-chave, linha do tempo, decisões, próximos passos, destaques).
    // Devolve { resumo, mudou }. Não altera o objeto recebido. A nota que o usuário escreve nunca passa por aqui.
    function trocarNoResumo(r, antigos, novo) {
        if (!r || typeof r !== 'object') return { resumo: r, mudou: false };
        let mudou = false;
        const t = s => { const v = trocar(s, antigos, novo); if (v !== s) mudou = true; return v; };
        const lista = (v, f) => Array.isArray(v) ? v.map(f) : v;
        const copia = { ...r };
        if (typeof r.resumo === 'string') copia.resumo = t(r.resumo);
        copia.topicos = lista(r.topicos, x => typeof x === 'string' ? t(x) : x);
        copia.decisoes = lista(r.decisoes, x => typeof x === 'string' ? t(x) : x);
        copia.capitulos = lista(r.capitulos, c => c && typeof c === 'object' ? { ...c, titulo: t(c.titulo), resumo: t(c.resumo) } : c);
        copia.acoes = lista(r.acoes, a => a && typeof a === 'object' ? { ...a, tarefa: t(a.tarefa), responsavel: t(a.responsavel) } : a);
        copia.destaques = lista(r.destaques, d => d && typeof d === 'object' ? { ...d, texto: t(d.texto) } : d);
        Object.keys(copia).forEach(k => { if (copia[k] === undefined) delete copia[k]; });
        return { resumo: copia, mudou };
    }

    // Resumo para exportar e arquivar: sem itens vazios (um Enter a mais ao editar) e sem linhas em branco sobrando. Não altera o original.
    function limparResumo(r) {
        if (!r || typeof r !== 'object') return r;
        const cheio = s => typeof s === 'string' && s.trim() !== '';
        const copia = { ...r };
        if (typeof r.resumo === 'string') copia.resumo = r.resumo.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean).join('\n\n');
        ['topicos', 'decisoes'].forEach(k => { if (Array.isArray(r[k])) copia[k] = r[k].filter(cheio).map(s => s.trim()); });
        if (Array.isArray(r.capitulos)) copia.capitulos = r.capitulos.filter(c => c && (cheio(c.titulo) || cheio(c.resumo)));
        if (Array.isArray(r.acoes)) copia.acoes = r.acoes.filter(a => a && cheio(a.tarefa));
        if (Array.isArray(r.destaques)) copia.destaques = r.destaques.filter(d => d && (cheio(d.marca) || cheio(d.texto)));
        return copia;
    }

    // Todos os nomes pelos quais este locutor já foi chamado: o rótulo, o nome atual e os anteriores (n.apelidos).
    function nomesDoRotulo(n, rotulo) {
        const atual = ((n.nomes || {})[rotulo] || '').trim();
        return [...new Set([rotulo, atual, ...((n.apelidos || {})[rotulo] || [])].filter(Boolean))];
    }

    // Qual locutor falava quando a pessoa digitou o comando ("Marina que está falando"). `t`: instante em que começou a digitar.
    // quando: 'agora' (quem falava em t, ou a última fala até então) ou 'anterior' (o locutor antes desse).
    // Devolve { estado: 'ok', rotulo } | { estado: 'ambiguo', candidatos: [rótulos] } | { estado: 'sem_fala' }.
    const TOLERANCIA_S = 1.5;    // a pessoa digita um instante depois de ouvir
    const ALCANCE_S = 15;        // sem fala em até 15 s antes: não há de quem falar
    const JANELA_AMBIGUA_S = 3;  // dois locutores nos últimos 3 s, cada um com pelo menos 25%: pergunta
    function acharLocutor(falas, t, quando) {
        const lista = (falas || []).filter(f => f && f.locutor && Number.isFinite(f.inicio) && Number.isFinite(f.fim) && f.inicio <= t + TOLERANCIA_S && f.fim >= t - ALCANCE_S)
            .sort((a, b) => a.inicio - b.inicio);
        if (!lista.length) return { estado: 'sem_fala' };
        const de = t - JANELA_AMBIGUA_S, ate = t + TOLERANCIA_S, total = ate - de;
        const cobertura = {};
        lista.forEach(f => { const o = Math.min(f.fim, ate) - Math.max(f.inicio, de); if (o > 0) cobertura[f.locutor] = (cobertura[f.locutor] || 0) + o; });
        if (quando !== 'anterior') {
            const fortes = Object.entries(cobertura).filter(([, o]) => o / total >= 0.25).sort((a, b) => b[1] - a[1]).map(([r]) => r);
            if (fortes.length > 1) return { estado: 'ambiguo', candidatos: fortes };
            if (fortes.length === 1) return { estado: 'ok', rotulo: fortes[0] };
            return { estado: 'ok', rotulo: lista[lista.length - 1].locutor };   // silêncio agora: a última fala até t
        }
        // anterior: o locutor que falou antes do que fala (ou falou por último) em t
        const atual = (Object.keys(cobertura).length ? Object.entries(cobertura).sort((a, b) => b[1] - a[1])[0][0] : lista[lista.length - 1].locutor);
        for (let i = lista.length - 1; i >= 0; i--) if (lista[i].locutor !== atual) return { estado: 'ok', rotulo: lista[i].locutor };
        return { estado: 'sem_fala' };
    }

    const api = { trocar, trocarNoResumo, limparResumo, nomesDoRotulo, acharLocutor };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else raiz.Nomes = api;
})(typeof window !== 'undefined' ? window : globalThis);
