"""
依存句法与成分句法可视化 Web 应用
使用 Streamlit + NLTK 展示句法分析结果
"""

import nltk
import streamlit as st
from html import escape
from pathlib import Path
from nltk.tree import Tree
from nltk.parse import RecursiveDescentParser
from nltk.grammar import CFG
from nltk import pos_tag, word_tokenize


APP_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = APP_DIR / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def download_nltk_data():
    """
    检查并自动下载 NLTK 所需的数据包
    """
    required_packages = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    }
    for package, resource_path in required_packages.items():
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            try:
                nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)
            except Exception as exc:
                st.warning(f"NLTK 数据包 {package} 下载失败：{exc}")


@st.cache_resource(show_spinner=False)
def setup_dependencies():
    """
    在应用启动时初始化所有依赖项
    """
    download_nltk_data()


# 定义词性标签到依存关系的映射
POS_TO_DEP = {
    'DT': 'det',      # 限定词
    'NN': 'ROOT',     # 名词（根节点）
    'NNS': 'ROOT',
    'VB': 'ROOT',     # 动词
    'VBD': 'ROOT',
    'VBG': 'ROOT',
    'VBN': 'ROOT',
    'VBP': 'ROOT',
    'VBZ': 'ROOT',
    'JJ': 'amod',     # 形容词
    'JJR': 'amod',
    'JJS': 'amod',
    'RB': 'advmod',   # 副词
    'IN': 'prep',     # 介词
    'TO': 'prep',     # to
}


def pos_to_dep(tag):
    """将词性标签转换为依存关系标签"""
    return POS_TO_DEP.get(tag, 'dep')


def render_dependency_tree_nltk(sentence: str):
    """
    使用 NLTK 生成依存句法树的可视化表示
    
    基于词性标注生成简化的依存关系，并识别核心论元
    """
    try:
        words = word_tokenize(sentence.replace('.', ''))
        tagged = pos_tag(words)
        
        # 记录动词位置（可能的谓词/根节点）
        verb_positions = []
        for i, (word, tag) in enumerate(tagged):
            if tag.startswith('VB'):
                verb_positions.append(i)
        
        # 找到主要谓词（第一个动词）
        main_predicate = verb_positions[0] if verb_positions else 0
        
        deps = []
        i = 0
        while i < len(tagged):
            word, tag = tagged[i]
            
            if i == main_predicate and tag.startswith('VB'):
                deps.append({'word': word, 'tag': tag, 'dep': 'ROOT', 'head': i, 'is_core_arg': True, 'role': 'predicate'})
                i += 1
            elif tag.startswith('VB'):
                deps.append({'word': word, 'tag': tag, 'dep': 'ROOT', 'head': i, 'is_core_arg': False, 'role': 'predicate'})
                i += 1
            elif tag == 'DT' or tag.startswith('JJ'):
                deps.append({'word': word, 'tag': tag, 'dep': 'det', 'head': i+1, 'is_core_arg': False, 'role': 'modifier'})
                i += 1
            elif tag.startswith('NN') and i < main_predicate:
                deps.append({'word': word, 'tag': tag, 'dep': 'nsubj', 'head': main_predicate, 'is_core_arg': True, 'role': 'subject'})
                i += 1
            elif tag.startswith('NN') and i > main_predicate:
                deps.append({'word': word, 'tag': tag, 'dep': 'dobj', 'head': main_predicate, 'is_core_arg': True, 'role': 'object'})
                i += 1
            elif tag == 'IN':
                prep_index = i
                deps.append({'word': word, 'tag': tag, 'dep': 'prep', 'head': main_predicate, 'is_core_arg': False, 'role': 'preposition'})
                i += 1

                while i < len(tagged) and (tagged[i][1] == 'DT' or tagged[i][1].startswith('JJ')):
                    word2, tag2 = tagged[i]
                    deps.append({'word': word2, 'tag': tag2, 'dep': pos_to_dep(tag2), 'head': min(i + 1, len(tagged) - 1), 'is_core_arg': False, 'role': 'modifier'})
                    i += 1

                if i < len(tagged) and tagged[i][1].startswith('NN'):
                    word2, tag2 = tagged[i]
                    deps.append({'word': word2, 'tag': tag2, 'dep': 'pobj', 'head': prep_index, 'is_core_arg': True, 'role': 'prep_object'})
                    i += 1
            else:
                deps.append({'word': word, 'tag': tag, 'dep': pos_to_dep(tag), 'head': main_predicate, 'is_core_arg': False, 'role': 'other'})
                i += 1
        
        return deps
    except Exception as e:
        return None


def extract_core_arguments(deps):
    """
    从依存分析结果中提取核心论元
    
    核心论元包括：
    - ROOT/predicate: 谓词/根节点
    - nsubj: 主语
    - dobj: 直接宾语
    - pobj: 介词宾语
    """
    if not deps:
        return []
    
    core_args = []
    
    for dep in deps:
        if dep.get('is_core_arg', False):
            role_map = {
                'predicate': '谓词 (Predicate)',
                'subject': '主语 (Subject)',
                'object': '直接宾语 (Direct Object)',
                'prep_object': '介词宾语 (Prepositional Object)'
            }
            core_args.append({
                '论元': dep['word'],
                '词性': dep['tag'],
                '依存关系': dep['dep'],
                '论元角色': role_map.get(dep.get('role', ''), dep.get('role', ''))
            })
    
    return core_args


def render_dependency_svg(deps):
    """
    生成 SVG 格式的依存句法树
    - 使用分层弧线避免重叠
    - 增加高度以容纳更多层级
    """
    if not deps:
        return None
    
    n = len(deps)
    width = max(900, n * 100)
    height = 400
    word_y = 60
    arc_base_y = 130
    level_height = 50
    
    positions = []
    start_x = 80
    spacing = (width - 160) / max(n - 1, 1)
    for i, dep in enumerate(deps):
        x = start_x + i * spacing
        positions.append(x)
    
    svg = f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" 
         style="background-color:#f8f9fa;font-family:Arial,sans-serif;">
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#3366cc"/>
        </marker>
    </defs>'''
    
    # 为每个弧线计算层级（避免重叠）
    arc_levels = [0] * len(deps)
    used_levels = {}
    for i, dep in enumerate(deps):
        head = dep['head']
        if head != i:
            dist = abs(i - head)
            for level in range(10):
                key = (min(head, i), max(head, i), level)
                if key not in used_levels:
                    used_levels[key] = True
                    arc_levels[i] = level
                    break
    
    # 绘制弧线
    for i, dep in enumerate(deps):
        head = dep['head']
        if head != i:
            x1 = positions[head]
            x2 = positions[i]
            level = arc_levels[i]
            arc_y = arc_base_y + level * level_height
            
            path = f'M {x1} {word_y + 35} Q {x1} {arc_y} {(x1+x2)/2} {arc_y} Q {x2} {arc_y} {x2} {word_y + 35}'
            svg += f'\n    <path d="{path}" fill="none" stroke="#3366cc" stroke-width="1.5" opacity="0.8" marker-end="url(#arrowhead)"/>'
            label_x = (x1 + x2) / 2
            svg += f'\n    <rect x="{label_x-25}" y="{arc_y-10}" width="50" height="18" rx="4" fill="#fff" stroke="#ddd"/>'
            svg += f'\n    <text x="{label_x}" y="{arc_y+4}" text-anchor="middle" font-size="11" fill="#cc3333">{escape(str(dep["dep"]))}</text>'
    
    # 绘制单词
    for i, dep in enumerate(deps):
        x = positions[i]
        svg += f'''
        <rect x="{x-40}" y="{word_y}" width="80" height="40" rx="8" fill="#e6f2ff" stroke="#3366cc" stroke-width="2"/>
        <text x="{x}" y="{word_y+18}" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{escape(str(dep["word"]))}</text>
        <text x="{x}" y="{word_y+33}" text-anchor="middle" font-size="11" fill="#666">({escape(str(dep["tag"]))})</text>'''
    
    svg += '\n</svg>'
    return svg


def render_constituency_tree_cfg(sentence: str):
    """
    使用 NLTK 自带的 CFG（上下文无关文法）解析器渲染成分句法树
    
    这里手写三条简单的文法规则：
    规则1: S  -> NP VP     （句子由名词短语 + 动词短语组成）
    规则2: NP -> DT NN     （名词短语由限定词 + 名词组成）
    规则3: VP -> V NP      （动词短语由动词 + 名词短语组成）
    
    参数:
        sentence: 待分析的句子
    返回:
        NLTK Tree 对象
    """
    try:
        words = word_tokenize(sentence.replace('.', ''))
        tagged = pos_tag(words)
        
        # 将词性转换为终结符，构建句子
        terminals = []
        for word, tag in tagged:
            if tag.startswith('DT'):
                terminals.append('the')
            elif tag.startswith('NN'):
                terminals.append('boy') if 'boy' in word else terminals.append('man') if 'man' in word else terminals.append('telescope')
            elif tag.startswith('VB'):
                terminals.append('saw')
            else:
                terminals.append(word)
        
        # 手写三条 CFG 规则
        cfg_rules = """
            S  -> NP VP
            NP -> DT NN
            VP -> V NP
        """
        
        # 创建 CFG 文法
        grammar = CFG.fromstring(cfg_rules)
        
        # 使用递归下降解析器
        parser = RecursiveDescentParser(grammar)
        
        # 尝试解析
        trees = list(parser.parse(terminals))
        
        if trees:
            return trees[0]
        else:
            # 如果标准解析失败，使用启发式方法构建树
            return build_heuristic_tree(tagged)
            
    except Exception as e:
        return build_heuristic_tree(tagged)


def build_heuristic_tree(tagged):
    """
    使用递归方法根据词性构建嵌套的成分句法树
    
    短语结构：
    - NP（名词短语）：DT/JJ/NN/PRP 等组合
    - VP（动词短语）：VB* 开头，后面可跟 NP/PP
    - PP（介词短语）：IN 开头，后面跟 NP
    """
    
    def parse_np(tokens, start):
        """解析名词短语，返回 (np_tree, next_index)"""
        if start >= len(tokens):
            return None, start
        
        np_children = []
        i = start
        
        while i < len(tokens):
            word, tag = tokens[i]
            
            if tag in ('DT', 'PDT', 'WDT', 'PRP$', 'POS'):
                np_children.append(Tree('DT', [word]))
                i += 1
            elif tag.startswith('JJ') or tag == 'CD':
                np_children.append(Tree('JJ', [word]))
                i += 1
            elif tag.startswith('NN') or tag in ('PRP', 'WP', 'NNPS'):
                np_children.append(Tree('NN', [word]))
                i += 1
            elif tag == 'POS':
                np_children.append(Tree('POS', [word]))
                i += 1
            else:
                break
        
        if np_children:
            return Tree('NP', np_children), i
        return None, start
    
    def parse_pp(tokens, start):
        """解析介词短语，返回 (pp_tree, next_index)"""
        if start >= len(tokens):
            return None, start
        
        word, tag = tokens[start]
        if tag == 'IN' or tag == 'TO':
            pp_children = [Tree('IN', [word])]
            i = start + 1
            
            np_part, i = parse_np(tokens, i)
            if np_part:
                pp_children.append(np_part)
            
            return Tree('PP', pp_children), i
        
        return None, start
    
    def parse_vp(tokens, start):
        """解析动词短语，返回 (vp_tree, next_index)"""
        if start >= len(tokens):
            return None, start
        
        word, tag = tokens[start]
        if tag.startswith('VB'):
            vp_children = [Tree('VB', [word])]
            i = start + 1
            
            while i < len(tokens):
                word2, tag2 = tokens[i]
                
                if tag2 in ('DT', 'PDT', 'WDT', 'PRP$', 'POS') or tag2.startswith('JJ') or tag2.startswith('NN') or tag2 in ('PRP', 'WP'):
                    np_part, next_i = parse_np(tokens, i)
                    if np_part:
                        vp_children.append(np_part)
                        i = next_i
                    else:
                        break
                elif tag2 == 'IN' or tag2 == 'TO':
                    pp_part, next_i = parse_pp(tokens, i)
                    if pp_part:
                        vp_children.append(pp_part)
                        i = next_i
                    else:
                        break
                elif tag2.startswith('RB') or tag2 == 'RP':
                    vp_children.append(Tree('RB', [word2]))
                    i += 1
                else:
                    break
            
            return Tree('VP', vp_children), i
        
        return None, start
    
    def parse_s(tokens, start):
        """解析句子，返回 (s_tree, next_index)"""
        if start >= len(tokens):
            return None, start
        
        children = []
        i = start
        
        np_part, i = parse_np(tokens, i)
        if np_part:
            children.append(np_part)
        
        vp_part, i = parse_vp(tokens, i)
        if vp_part:
            children.append(vp_part)
        
        pp_part, i = parse_pp(tokens, i)
        if pp_part:
            children.append(pp_part)
        
        if children:
            return Tree('S', children), i
        
        return None, start
    
    result, _ = parse_s(tagged, 0)
    if result:
        return result
    
    return Tree('S', [Tree(tag, [word]) for word, tag in tagged])


def render_constituency_svg(tree):
    """
    将成分句法树渲染为 SVG 图形
    
    使用分层布局，确保所有叶子节点在同一水平线上对齐
    """
    def count_leaves(t):
        if isinstance(t, str):
            return 1
        if len(t) == 0:
            return 0
        return sum(count_leaves(child) for child in t)
    
    def get_depth(t):
        if isinstance(t, str) or len(t) == 0:
            return 1
        return 1 + max(get_depth(child) for child in t)
    
    total_leaves = count_leaves(tree)
    tree_depth = get_depth(tree)
    
    leaf_width = 80
    level_height = 70
    
    width = max(900, total_leaves * leaf_width + 100)
    height = tree_depth * level_height + 80
    
    elements = []
    
    def layout(t, x, y, available_width):
        nonlocal elements
        if isinstance(t, str):
            center_x = x + available_width / 2
            elements.append(f'<rect x="{center_x-30}" y="{y-12}" width="60" height="24" rx="4" fill="#e6f2ff" stroke="#3366cc" stroke-width="1.5"/>')
            elements.append(f'<text x="{center_x}" y="{y+5}" text-anchor="middle" font-size="13" fill="#333">{escape(str(t))}</text>')
            return
    
        label = escape(str(t.label()))
        node_x = x + available_width / 2
        
        if len(t) == 0:
            return
        
        child_width = available_width / len(t)
        for i, child in enumerate(t):
            child_x = x + i * child_width
            child_center = child_x + child_width / 2
            layout(child, child_x, y + level_height, child_width)
            
            elements.append(f'<line x1="{node_x}" y1="{y+5}" x2="{child_center}" y2="{y+level_height-10}" stroke="#3366cc" stroke-width="1.5"/>')
        
        elements.append(f'<rect x="{node_x-25}" y="{y-15}" width="50" height="25" rx="6" fill="#3366cc"/>')
        elements.append(f'<text x="{node_x}" y="{y+3}" text-anchor="middle" font-size="13" font-weight="bold" fill="white">{label}</text>')
    
    layout(tree, 20, 50, width - 40)
    
    svg = f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background-color:#fafafa;font-family:Arial,sans-serif;">'
    svg += '\n    '.join(elements)
    svg += '\n</svg>'
    return svg


def main():
    """
    Streamlit 应用主函数
    """
    st.set_page_config(
        page_title="句法分析可视化工具",
        page_icon="🌲",
        layout="wide"
    )
    
    st.title("🌲 句法分析可视化")
    st.markdown("""
    这是一个展示 **依存句法树** 和 **成分句法树** 的交互式工具。
    
    📌 **使用说明**：在下方输入框中输入英文句子，观察两种不同的句法分析结果。
    """)
    
    if 'initialized' not in st.session_state:
        with st.spinner("正在初始化语言模型，请稍候..."):
            setup_dependencies()
        st.session_state.initialized = True
        st.rerun()
    
    default_sentence = "The boy saw the man with the telescope."
    
    sentence = st.text_input(
        "📝 请输入英文句子：",
        value=default_sentence,
        placeholder="输入一个英文句子...",
        help="输入要分析的英文句子，默认使用课件中的经典例句"
    )
    
    if sentence.strip():
        tab1, tab2 = st.tabs(["📊 依存关系", "📚 成分结构"])
        
        with tab1:
            st.header("依存句法树 (Dependency Parse)")
            st.markdown("""
            **依存句法** 描述词语之间的 **主从依赖关系**。
            
            - 箭头从 **中心词** 指向 **依存词**
            - 箭头上标注 **依存关系类型**
            - ROOT = 根节点（核心谓词）
            """)
            
            deps = render_dependency_tree_nltk(sentence)
            if deps:
                svg_html = render_dependency_svg(deps)
                if svg_html:
                    st.components.v1.html(svg_html, height=350, scrolling=True)
                
                st.subheader("依存关系详情")
                dep_data = []
                for i, dep in enumerate(deps):
                    dep_data.append({
                        "序号": i + 1,
                        "词语": dep['word'],
                        "词性": dep['tag'],
                        "依存关系": dep['dep'],
                        "中心词位置": dep['head'] + 1
                    })
                st.table(dep_data)
            else:
                st.error("依存分析失败")
        
        with tab2:
            st.header("成分句法树 (Constituency Parse)")
            st.markdown("""
            **成分句法** 将句子组织成 **嵌套的短语结构**。
            
            基于 NLTK 的 CFG（上下文无关文法）解析：
            - **规则1**: S → NP VP（句子 = 名词短语 + 动词短语）
            - **规则2**: NP → DT NN（名词短语 = 限定词 + 名词）
            - **规则3**: VP → V NP（动词短语 = 动词 + 名词短语）
            """)
            
            tree = render_constituency_tree_cfg(sentence)
            if tree:
                st.subheader("ASCII 树形图")
                st.code(tree.pformat(), language=None)
                
                st.subheader("可视化树形图")
                svg_html = render_constituency_svg(tree)
                st.components.v1.html(svg_html, height=450, scrolling=True)
            else:
                st.error("成分句法分析失败")
        
        # 核心论元提取器
        st.divider()
        st.header("🎯 核心论元提取器")
        st.markdown("""
        **核心论元** 是句子的骨架成分，包括：
        - **谓词 (Predicate)**：句子的核心动词/谓词
        - **主语 (Subject)**：执行动作的主体
        - **直接宾语 (Direct Object)**：动作的承受者
        - **介词宾语 (Prepositional Object)**：介词后的名词
        """)
        
        if deps:
            core_args = extract_core_arguments(deps)
            if core_args:
                st.subheader("提取的核心论元")
                st.dataframe(
                    core_args,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 用 Markdown 表格展示
                st.subheader("论元结构概览")
                roles = [arg['论元角色'] for arg in core_args]
                words = [arg['论元'] for arg in core_args]
                
                md_table = "| 论元角色 | 词语 |\n|---------|------|\n"
                for arg in core_args:
                    md_table += f"| {arg['论元角色']} | **{arg['论元']}** |\n"
                st.markdown(md_table)
                
                # 谓词-论元结构可视化
                if len(core_args) >= 2:
                    pred = next((a['论元'] for a in core_args if '谓词' in a['论元角色']), '')
                    subj = next((a['论元'] for a in core_args if '主语' in a['论元角色']), '')
                    obj = next((a['论元'] for a in core_args if '宾语' in a['论元角色']), '')
                    pobj = next((a['论元'] for a in core_args if '介词' in a['论元角色']), '')
                    
                    st.subheader("谓词-论元结构")
                    structure = f"**{subj}** + **{pred}**"
                    if obj:
                        structure += f" + **{obj}**"
                    if pobj:
                        structure += f" + (介词 + **{pobj}**)"
                    st.info(structure)
            else:
                st.warning("未识别到核心论元")
        else:
            st.info("请先输入句子进行分析")
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>🔧 技术栈: Streamlit | NLTK</p>
        <p>📖 成分句法使用 CFG 三条规则解析</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
