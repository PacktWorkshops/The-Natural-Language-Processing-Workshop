# # This notebook will generate word dependency tree  and text entities
# ### To download `en_core_web_sm` run this command in your virtual environment :-
# `python -m spacy download en_core_web_sm`

import spacy
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()

doc = nlp('God helps those who help themselves.')
displacy.render(doc, style='dep', jupyter=True)

text = 'Once upon a time there lived a saint named Ramakrishna Paramahansa.         His chief disciple Narendranath Dutta also known as Swami Vivekananda         is the founder of Ramakrishna Mission and Ramakrishna Math.'
doc2 = nlp(text)
displacy.render(doc2, style='ent', jupyter=True)
