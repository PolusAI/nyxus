#pragma once

// Mechanics of whole-slide detection: the label directory names the intensity directory. Asserts
// which mode the directory arguments select, not any feature value. Claims no oracle (SPEC 2).

#include <gtest/gtest.h>
#include <chrono>
#include <fstream>

#include "../src/nyx/dirs_and_files.h"		// also supplies <optional>, <string>, <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/helpers/fsystem.h"

// A fresh directory "wsi" and a sibling "other" under the temp directory, removed on scope exit
struct WholeslideDirs
{
	fs::path root, wsi, other;

	WholeslideDirs()
	{
		root = fs::temp_directory_path() / ("nyxus_wholeslide_dirs_" + std::to_string (std::chrono::steady_clock::now().time_since_epoch().count()));
		wsi = root / "wsi";
		other = root / "other";
		fs::create_directories (wsi);
		fs::create_directories (other);
	}
	~WholeslideDirs()
	{
		std::error_code ec;
		fs::remove_all (root, ec);
	}
};

// Spellings of one directory that differ as strings: a trailing separator of either kind, a trailing
// ".", a ".." round trip, and a symbolic link where this host lets us make one
static std::vector<std::string> wholeslide_dirs_spellings (const WholeslideDirs& t)
{
	const std::string d = t.wsi.string();
	std::vector<std::string> v = { d + "/", d + std::string (1, (char) fs::path::preferred_separator), d + "/.",
		(t.root / "other" / ".." / "wsi").string() };

	std::error_code ec;
	fs::create_directory_symlink (t.wsi, t.root / "wsi_link", ec);
	if (!ec)
		v.push_back ((t.root / "wsi_link").string());

	return v;
}

// Parses a command line naming the two directories; the output directory is the temp directory
static bool assert_wholeslide_dirs_parse (Environment& e, const std::string& intdir, const std::string& segdir, std::string* err = nullptr)
{
	std::vector<std::string> args = { "nyxus", "--intDir=" + intdir, "--segDir=" + segdir,
		"--outDir=" + fs::temp_directory_path().string(), "--features=MEAN", "--filePattern=.*\\.tif",
		"--outputType=singlecsv" };
	std::vector<char*> argv;
	for (auto& a : args)
		argv.push_back (a.data());
	testing::internal::CaptureStderr();
	bool ok = e.parse_cmdline ((int) argv.size(), argv.data());
	std::string captured = testing::internal::GetCapturedStderr();
	if (err)
		*err = captured;
	return ok;
}

// Every spelling of a directory is that directory, and a sibling is not
void test_wholeslide_dirs_same_directory_spellings_mechanics()
{
	WholeslideDirs t;
	const std::string d = t.wsi.string();

	bool same = false;
	ASSERT_FALSE (Nyxus::same_directory (d, d, same).has_value());
	ASSERT_TRUE (same);

	for (const auto& s : wholeslide_dirs_spellings (t))
	{
		SCOPED_TRACE (s);
		same = false;
		ASSERT_FALSE (Nyxus::same_directory (d, s, same).has_value());
		ASSERT_TRUE (same);
	}

	same = true;
	ASSERT_FALSE (Nyxus::same_directory (d, t.other.string(), same).has_value());
	ASSERT_FALSE (same);
}

// The CLI selects whole-slide mode for every spelling of the intensity directory, and segmented mode
// for a different directory
void test_wholeslide_dirs_cli_any_spelling_is_wholeslide_mechanics()
{
	WholeslideDirs t;
	const std::string d = t.wsi.string();

	for (const auto& s : wholeslide_dirs_spellings (t))
	{
		SCOPED_TRACE (s);
		Environment e;
		std::string err;
		ASSERT_TRUE (assert_wholeslide_dirs_parse (e, d, s, &err)) << err;
		ASSERT_TRUE (e.singleROI);
	}

	Environment e;
	std::string err;
	ASSERT_TRUE (assert_wholeslide_dirs_parse (e, d, t.other.string(), &err)) << err;
	ASSERT_FALSE (e.singleROI);
}

// Two directories whose names differ only in case are two directories on a case-sensitive
// filesystem, and the CLI runs segmented. Where the filesystem folds case they are one directory and
// there is nothing to test.
void test_wholeslide_dirs_cli_case_differing_dirs_stay_segmented_mechanics()
{
	WholeslideDirs t;
	const fs::path upper = t.root / "WSI";
	if (fs::exists (upper))
		GTEST_SKIP() << "this filesystem folds case: WSI and wsi are one directory";
	fs::create_directory (upper);

	Environment e;
	std::string err;
	ASSERT_TRUE (assert_wholeslide_dirs_parse (e, t.wsi.string(), upper.string(), &err)) << err;
	ASSERT_FALSE (e.singleROI);
}

// The refusal: a directory that is not there, or is a file, cannot be compared with the other one, so
// the command line is refused rather than read as two directories
void test_wholeslide_dirs_missing_dir_refused_mechanics()
{
	WholeslideDirs t;
	const std::string d = t.wsi.string(),
		missing = (t.root / "missing").string(),
		file = (t.root / "file.tif").string();
	{ std::ofstream f (file); f << "x"; }

	for (const auto& bad : { missing, file })
	{
		SCOPED_TRACE (bad);
		bool same = true;
		auto ermsg = Nyxus::same_directory (d, bad, same);
		ASSERT_TRUE (ermsg.has_value());
		EXPECT_NE (ermsg->find ("cannot access directory " + bad), std::string::npos) << *ermsg;
		ermsg = Nyxus::same_directory (bad, d, same);
		ASSERT_TRUE (ermsg.has_value());
		EXPECT_NE (ermsg->find ("cannot access directory " + bad), std::string::npos) << *ermsg;

		for (bool badIsSeg : { true, false })
		{
			Environment e;
			std::string err;
			ASSERT_FALSE (badIsSeg ? assert_wholeslide_dirs_parse (e, d, bad, &err) : assert_wholeslide_dirs_parse (e, bad, d, &err));
			EXPECT_NE (err.find ("cannot access directory " + bad), std::string::npos) << err;
		}
	}
}
