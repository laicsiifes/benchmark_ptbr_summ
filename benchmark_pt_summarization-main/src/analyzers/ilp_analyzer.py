
import pulp


def select_sentences(sentences: list, summary_size: int, excluded_solutions: list = None) -> list:

    dict_concepts_sentences = {}

    for sentence in sentences:
        if sentence.concepts is not None:
            for concept in sentence.concepts:
                concept_sentences = []
                if concept in dict_concepts_sentences:
                    concept_sentences = dict_concepts_sentences[concept]
                concept_sentences.append(sentence)
                dict_concepts_sentences[concept] = concept_sentences

    id_concept = 1

    objetive_function = None
    dict_pulp_variables = {}

    for concept in dict_concepts_sentences.keys():
        if concept.weight > 0:
            concept.id_token = id_concept
            label = 'c_' + str(id_concept)
            c = pulp.LpVariable(label, lowBound=0, upBound=1, cat='Binary')
            dict_pulp_variables[label] = c
            if id_concept > 1:
                objetive_function += (concept.weight * c)
            else:
                objetive_function = (concept.weight * c)
            id_concept += 1

    ilp_problem = pulp.LpProblem('ilp_model', pulp.LpMaximize)

    ilp_problem += objetive_function

    sentence = sentences[0]

    id_sentence = 's_' + str(sentence.id_sentence)

    s = pulp.LpVariable(id_sentence, lowBound=0, upBound=1, cat='Binary')

    dict_pulp_variables[id_sentence] = s

    sentence_size_constrant = (len([t for t in sentence.tokens if t.token_type == 'WORD' or t.token_type == 'NUM']) * s)

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        id_sentence = 's_' + str(sentence.id_sentence)
        s = pulp.LpVariable(id_sentence, lowBound=0, upBound=1, cat='Binary')
        sentence_size = len([t for t in sentence.tokens if t.token_type == 'WORD' or t.token_type == 'NUM'])
        sentence_size_constrant += (sentence_size * s)
        dict_pulp_variables[id_sentence] = s

    ilp_problem.addConstraint(sentence_size_constrant <= summary_size)

    for concept, concept_sentences in dict_concepts_sentences.items():
        if concept.weight > 0:
            label_concept = 'c_' + str(concept.id_token)
            c = dict_pulp_variables[label_concept]
            concept_all_sentences_constrant = -1 * c
            for sentence in concept_sentences:
                id_sentence = 's_' + str(sentence.id_sentence)
                s = dict_pulp_variables[id_sentence]
                concept_sentence_constrant = -1 * c + 1 * s
                ilp_problem.addConstraint(concept_sentence_constrant <= 0)
                concept_all_sentences_constrant += 1 * s
            ilp_problem.addConstraint(concept_all_sentences_constrant >= 0)

    if excluded_solutions is not None and len(excluded_solutions) > 0:
        for excluded_solution in excluded_solutions:
            s = dict_pulp_variables[excluded_solution[0]]
            excluded_solution_constrant = s
            lenght = len(excluded_solution)
            for i in range(1, lenght):
                s = dict_pulp_variables[excluded_solution[i]]
                excluded_solution_constrant += s
            ilp_problem.addConstraint(excluded_solution_constrant <= (lenght - 2))

    ilp_problem.solve(pulp.GLPK(msg=False))

    selected_variables = []

    for variable in ilp_problem.variables():
        if variable.varValue == 1:
            selected_variables.append(variable.name)

    summary_sentences = []

    for sentence in sentences:
        id_sentence = 's_' + str(sentence.id_sentence)
        if id_sentence in selected_variables:
            summary_sentences.append(sentence)

    return summary_sentences
