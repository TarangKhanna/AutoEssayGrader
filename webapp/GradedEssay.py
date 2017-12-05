class GradedEssay:
	def __init__(self, et, eg, efn, cs, wc, swc, sec, se, gil, gic):
		self.essay_text = et
		self.essay_grade = eg
		self.essay_file_name = efn
		self.confidence_score = cs
		self.word_count = wc
		self.stop_word_count = swc
		self.spelling_error_count = sec
		self.spelling_errors = se
		self.grammar_issues_list = gil
		self.grammar_issues_count = gic
