#pragma once

class ResultOptions
{
public:

	// getters
	inline double noval() const { return noval_; }
	inline double tiny() const { return tiny_; }
	inline bool need_aggregation() const { return need_aggregation_; }
	inline bool need_annotation() const { return need_annotation_; }
	inline std::string anno_separator() const { return anno_sep_; }

	// setters (python api scenario)
	inline void set_noval (double a) { noval_ = a; }
	inline void set_tiny (double a) { tiny_ = a; }
	inline void set_need_aggregation (bool a) { need_aggregation_ = a; }
	inline void set_need_annotation (bool a) { need_annotation_ = a; }
	inline void set_anno_separator (const std::string & a) { anno_sep_ = a; }

	// raw inputs (cli scenario)
	std::string raw_noval,
		raw_tiny,
		raw_aggregate,
		raw_annotate,
		raw_anno_separator;

	// true if the parameters have never been specified via raw_<whatever>
	bool nothing2parse() { return raw_noval.empty() && raw_tiny.empty() && raw_aggregate.empty() && raw_annotate.empty() && raw_anno_separator.empty(); }

	std::tuple<bool, std::optional<std::string>> parse_input()
	{
		float val;

		if (!raw_noval.empty())
		{
			if (!Nyxus::parse_as_float(raw_noval, val))
				return { false, "Error in " + raw_noval + ": expecting a real value" };
			set_noval(val);
		}

		if (!raw_tiny.empty())
		{
			if (!Nyxus::parse_as_float(raw_tiny, val))
				return { false, "Error in " + raw_tiny + ": expecting a real value" };
			set_tiny(val);
		}

		if (!raw_aggregate.empty())
		{
			bool bval;
			if (!Nyxus::parse_as_bool(raw_aggregate, bval))
				return { false, "Error in " + raw_aggregate + ": expecting a boolean constant 'true' or 'false'" };
			set_need_aggregation (bval);
		}

		if (!raw_annotate.empty())
		{
			bool bval;
			if (!Nyxus::parse_as_bool(raw_annotate, bval))
				return { false, "Error in " + raw_annotate + ": expecting a boolean constant 'true' or 'false'" };
			set_need_annotation (bval);
		}

		if (!raw_anno_separator.empty())
		{
			set_anno_separator (raw_anno_separator);
		}

		return { true, std::nullopt };
	}

private:

	double noval_ = 0.0,
		tiny_ = 1e-10;
	bool need_aggregation_ = false,
		need_annotation_ = false;
	std::string anno_sep_ = "_";
};

