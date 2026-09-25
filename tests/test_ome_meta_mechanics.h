#pragma once

// Unit tests for the native-OME metadata parsers / OmeAxes descriptor.
// The OME-XML parser and OmeAxes helpers are exercised here with embedded
// fixtures (no external files). The OME-Zarr parser is guarded by
// OMEZARR_SUPPORT (compiled only in USE_Z5 builds); it is additionally
// validated end-to-end against the reference-data matrix by
// io_oracle/ome_meta_selftest.cpp.

#include <gtest/gtest.h>
#include "../src/nyx/ome/ome_axes.h"
#include "../src/nyx/ome/ome_tiff_meta.h"

// 5D, non-default DimensionOrder XYZTC (C and T swapped) — the case that catches
// a reader assuming TCZYX. Taken verbatim from io_oracle/fixtures/xml/5d_ctzyx.xml.
static const char* kOmeXml_5d_ctzyx =
	"<?xml version=\"1.0\" encoding=\"UTF-8\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">"
	"<Image ID=\"Image:0\" Name=\"Image0\"><Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZTC\" Type=\"uint16\" "
	"SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\" SizeT=\"2\" SizeC=\"3\" "
	"PhysicalSizeX=\"0.5\" PhysicalSizeXUnit=\"micrometer\" PhysicalSizeY=\"0.5\" PhysicalSizeYUnit=\"micrometer\" "
	"PhysicalSizeZ=\"2.0\" PhysicalSizeZUnit=\"micrometer\" TimeIncrement=\"1.0\" TimeIncrementUnit=\"second\">"
	"<Channel ID=\"Channel:0:0\" Name=\"DAPI\" SamplesPerPixel=\"1\"/>"
	"<Channel ID=\"Channel:0:1\" Name=\"GFP\" SamplesPerPixel=\"1\"/>"
	"<TiffData IFD=\"0\" PlaneCount=\"24\"/></Pixels></Image></OME>";

static const char* kOmeXml_2d =
	"<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"><Image ID=\"Image:0\">"
	"<Pixels ID=\"Pixels:0\" DimensionOrder=\"XYCZT\" Type=\"uint16\" SizeX=\"8\" SizeY=\"6\" SizeZ=\"1\" "
	"SizeC=\"1\" SizeT=\"1\" PhysicalSizeX=\"0.5\" PhysicalSizeY=\"0.5\"></Pixels></Image></OME>";

TEST(OmeTiffMeta, ParsesNonDefault5D)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(kOmeXml_5d_ctzyx);
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.sizeY, 6u);
	EXPECT_EQ(a.sizeZ, 4u);
	EXPECT_EQ(a.sizeC, 3u);
	EXPECT_EQ(a.sizeT, 2u);
	// XYZTC means C slowest, T next -> on-disk (slowest-first) is C,T,Z,Y,X.
	EXPECT_EQ(a.storageOrder, "CTZYX");
	EXPECT_EQ(a.omeDimensionOrder, "XYZTC");
	EXPECT_EQ(a.dtype, Nyxus::PixelType::UInt16);
	EXPECT_EQ(a.bitsPerSample, 16u);
	EXPECT_DOUBLE_EQ(a.physX, 0.5);
	EXPECT_DOUBLE_EQ(a.physZ, 2.0);
	EXPECT_EQ(a.unitZ, "micrometer");
	// axis-role lookup independent of on-disk order
	EXPECT_EQ(a.storageIndexOf('C'), 0);   // C is first (slowest) in CTZYX
	EXPECT_EQ(a.storageIndexOf('X'), 4);
}

// <TiffData> plane->IFD mapping. Base XML: XYZCT, Z=2 C=2 T=1 -> 4 planes, canonical
// ordinal ord = z + c*2.
static std::string tiffdata_xml(const std::string& blocks)
{
	return "<OME><Image><Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
	       "SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">" + blocks +
	       "</Pixels></Image></OME>";
}

// A single canonical block (tifffile's form) means contiguous-from-IFD-0 == the default, so
// the map is left empty and ifdForPlane stays canonical.
TEST(OmeTiffMetaTiffData, CanonicalSingleBlockStaysIdentity)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(tiffdata_xml("<TiffData IFD=\"0\" PlaneCount=\"4\"/>"));
	ASSERT_TRUE(a.valid);
	EXPECT_TRUE(a.planeToIfd.empty());
	EXPECT_FALSE(a.multiFileTiff);
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 0u);
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 3u);   // ord = 1 + 1*2
}

// Per-plane blocks that reverse the IFDs: ifdForPlane must return the declared IFD.
TEST(OmeTiffMetaTiffData, ReversedPerPlaneMapping)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(tiffdata_xml(
		"<TiffData FirstC=\"0\" FirstZ=\"0\" FirstT=\"0\" IFD=\"3\" PlaneCount=\"1\"/>"
		"<TiffData FirstC=\"0\" FirstZ=\"1\" FirstT=\"0\" IFD=\"2\" PlaneCount=\"1\"/>"
		"<TiffData FirstC=\"1\" FirstZ=\"0\" FirstT=\"0\" IFD=\"1\" PlaneCount=\"1\"/>"
		"<TiffData FirstC=\"1\" FirstZ=\"1\" FirstT=\"0\" IFD=\"0\" PlaneCount=\"1\"/>"));
	ASSERT_TRUE(a.valid);
	ASSERT_EQ(a.planeToIfd.size(), 4u);
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 3u);
	EXPECT_EQ(a.ifdForPlane(1, 0, 0), 2u);
	EXPECT_EQ(a.ifdForPlane(0, 1, 0), 1u);
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 0u);
}

// P5: PlaneCount omitted -> defaults to "the remaining planes from the start plane" (OME
// spec). A single block with only IFD given must map every plane from that offset.
TEST(OmeTiffMetaTiffData, OmittedPlaneCountCoversRemaining)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(tiffdata_xml("<TiffData IFD=\"5\"/>"));
	ASSERT_TRUE(a.valid);
	ASSERT_EQ(a.planeToIfd.size(), 4u);          // 4 planes (Z=2,C=2)
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 5u);       // start at IFD 5
	EXPECT_EQ(a.ifdForPlane(1, 0, 0), 6u);
	EXPECT_EQ(a.ifdForPlane(0, 1, 0), 7u);
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 8u);       // all four planes mapped, consecutively
}

// A non-zero starting IFD (planes contiguous but offset, e.g. a multi-image container).
TEST(OmeTiffMetaTiffData, NonZeroStartOffsetsAllPlanes)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(tiffdata_xml("<TiffData IFD=\"10\" PlaneCount=\"4\"/>"));
	ASSERT_TRUE(a.valid);
	ASSERT_EQ(a.planeToIfd.size(), 4u);
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 10u);
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 13u);
}

// A <TiffData> naming another file (<UUID> child) is multi-file: unsupported, flagged, and
// that plane is left at the canonical fallback rather than pointed at a wrong local IFD.
TEST(OmeTiffMetaTiffData, MultiFileUuidBlockFlaggedAndSkipped)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(tiffdata_xml(
		"<TiffData FirstC=\"0\" FirstZ=\"0\" FirstT=\"0\" IFD=\"0\">"
		"<UUID FileName=\"other.tif\">urn:uuid:abc</UUID></TiffData>"));
	ASSERT_TRUE(a.valid);
	EXPECT_TRUE(a.multiFileTiff);
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 0u);   // canonical fallback (not mapped)
}

// A <UUID> naming the file's own UUID (root <OME UUID=...>) is this file, as Bio-Formats writes
// every block of a single-file OME-TIFF: the block maps its planes and the file is not
// multi-file. A different UUID in another block of the same file still is. The UUID says so
// whatever the file is called on disk, so these pass the name of a file that is NOT the one the
// blocks record -- an empty name is not a case any caller produces, since read_ome_tiff_axes
// always hands the parser TIFFFileName().
TEST(OmeTiffMetaTiffData, OwnUuidBlockMapsAndIsNotMultiFile)
{
	auto xml = [](const std::string& blocks)
	{
		return "<OME UUID=\"urn:uuid:self\"><Image><Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
		       "SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">" + blocks + "</Pixels></Image></OME>";
	};

	Nyxus::OmeAxes own = Nyxus::parse_ome_xml(xml(
		"<TiffData FirstZ=\"0\" FirstC=\"0\" FirstT=\"0\" IFD=\"4\" PlaneCount=\"4\">"
		"<UUID FileName=\"this.ome.tif\"> urn:uuid:self </UUID></TiffData>"),
		"/archive/renamed-copy.ome.tif");
	ASSERT_TRUE(own.valid);
	EXPECT_FALSE(own.multiFileTiff);
	EXPECT_EQ(own.ifdForPlane(0, 0, 0), 4u);
	EXPECT_EQ(own.ifdForPlane(1, 1, 0), 7u);

	Nyxus::OmeAxes mixed = Nyxus::parse_ome_xml(xml(
		"<TiffData FirstZ=\"0\" FirstC=\"0\" FirstT=\"0\" IFD=\"0\" PlaneCount=\"2\"><UUID>urn:uuid:self</UUID></TiffData>"
		"<TiffData FirstZ=\"0\" FirstC=\"1\" FirstT=\"0\" IFD=\"0\" PlaneCount=\"2\"><UUID>urn:uuid:other</UUID></TiffData>"),
		"this.ome.tif");
	ASSERT_TRUE(mixed.valid);
	EXPECT_TRUE(mixed.multiFileTiff);
}

// Regression: renaming or copying a single-file OME-TIFF must not turn it into a "multi-file" one.
// The FileName attribute records what the file was called when it was written, so on a renamed
// file it disagrees with the name the reader is holding -- while the UUID, which identifies the
// file itself, still agrees with the root's. The UUID therefore decides wherever there is a pair
// to compare, and the name only when there is not. What this discriminates: a reader that lets a
// FileName mismatch override a matching UUID throws on every renamed single-file OME-TIFF, which
// is most of the ones that have been through a pipeline.
TEST(OmeTiffMetaTiffData, RenamedSingleFileIsNotMultiFile)
{
	const std::string xml =
		"<OME UUID=\"urn:uuid:self\"><Image><Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
		"SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">"
		"<TiffData FirstZ=\"0\" FirstC=\"0\" FirstT=\"0\" IFD=\"4\" PlaneCount=\"4\">"
		"<UUID FileName=\"as-written.ome.tif\">urn:uuid:self</UUID></TiffData>"
		"</Pixels></Image></OME>";

	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(xml, "/data/run17/renamed.ome.tif");
	ASSERT_TRUE(a.valid);
	EXPECT_FALSE(a.multiFileTiff) << "a stale FileName must not outvote the file's own UUID";
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 4u) << "and the block still maps its planes";
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 7u);

	// the same document read from the file it was written as: unchanged verdict
	Nyxus::OmeAxes same = Nyxus::parse_ome_xml(xml, "as-written.ome.tif");
	EXPECT_FALSE(same.multiFileTiff);
	EXPECT_EQ(same.ifdForPlane(0, 0, 0), 4u);
}

// What marks a companion file is the <UUID FileName=...> attribute, so a single-file OME-TIFF is
// read whether or not the reader can compare UUIDs: the root UUID is optional in the schema, the
// root element may carry a namespace prefix, and a <UUID/> may be self-closing or name this very
// file. What this discriminates: treating every <UUID> child as a companion rejects ordinary
// single-file OME-TIFFs outright, and searching for </UUID> past this block's </TiffData> picks up
// a later block's text.
TEST(OmeTiffMetaTiffData, SingleFileUuidFormsAreNotMultiFile)
{
	auto pixels = [](const std::string& root_attrs, const std::string& blocks)
	{
		return "<" + root_attrs + "><Image><Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
		       "SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">" + blocks + "</Pixels></Image></"
		       + root_attrs.substr(0, root_attrs.find(' ')) + ">";
	};

	// no root UUID, and the block names this file
	Nyxus::OmeAxes named = Nyxus::parse_ome_xml(pixels("OME",
		"<TiffData IFD=\"0\" PlaneCount=\"4\"><UUID FileName=\"self.ome.tif\">urn:uuid:a</UUID></TiffData>"),
		"/data/SELF.ome.tif");
	ASSERT_TRUE(named.valid);
	EXPECT_FALSE(named.multiFileTiff) << "a block naming this very file is not a companion";

	// namespace-prefixed root, and the block names this file
	Nyxus::OmeAxes prefixed = Nyxus::parse_ome_xml(pixels("ome:OME xmlns:ome=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"",
		"<TiffData IFD=\"0\" PlaneCount=\"4\"><UUID FileName=\"self.ome.tif\">urn:uuid:a</UUID></TiffData>"),
		"self.ome.tif");
	ASSERT_TRUE(prefixed.valid);
	EXPECT_FALSE(prefixed.multiFileTiff);

	// a self-closing <UUID/> names no file, so the planes are this file's -- and the </UUID> of a
	// LATER block must not be read as this one's text
	Nyxus::OmeAxes selfclosed = Nyxus::parse_ome_xml(pixels("OME UUID=\"urn:uuid:self\"",
		"<TiffData FirstZ=\"0\" IFD=\"0\" PlaneCount=\"2\"><UUID/></TiffData>"
		"<TiffData FirstZ=\"1\" IFD=\"2\" PlaneCount=\"2\"><UUID>urn:uuid:self</UUID></TiffData>"),
		"self.ome.tif");
	ASSERT_TRUE(selfclosed.valid);
	EXPECT_FALSE(selfclosed.multiFileTiff);

	// and the genuine companion is still caught, by file name
	Nyxus::OmeAxes companion = Nyxus::parse_ome_xml(pixels("OME",
		"<TiffData IFD=\"0\" PlaneCount=\"4\"><UUID FileName=\"other.ome.tif\">urn:uuid:b</UUID></TiffData>"),
		"self.ome.tif");
	ASSERT_TRUE(companion.valid);
	EXPECT_TRUE(companion.multiFileTiff);
}

// OME-XML may carry its namespace prefix on every element, not only the root: Bio-Formats writes
// unprefixed documents, but a document that has been through an XML toolchain often comes back as
// <ome:OME><ome:Image><ome:Pixels>. What this discriminates: matching only the literal "<Pixels"
// and "<TiffData" reads such a file as having no OME metadata at all, so a 5D volume silently
// degrades to a plain Z-stack with a canonical plane->IFD map.
TEST(OmeTiffMetaTiffData, PrefixedChildElementsParse)
{
	const std::string prefixed =
		"<ome:OME xmlns:ome=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" UUID=\"urn:uuid:self\">"
		"<ome:Image><ome:Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
		"SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">"
		"<ome:TiffData FirstZ=\"0\" FirstC=\"0\" FirstT=\"0\" IFD=\"4\" PlaneCount=\"4\">"
		"<ome:UUID FileName=\"self.ome.tif\">urn:uuid:self</ome:UUID></ome:TiffData>"
		"</ome:Pixels></ome:Image></ome:OME>";

	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(prefixed, "self.ome.tif");
	ASSERT_TRUE(a.valid) << "a fully prefixed document is still OME-XML";
	EXPECT_EQ(a.sizeZ, 2u);
	EXPECT_EQ(a.sizeC, 2u);
	EXPECT_FALSE(a.multiFileTiff);
	EXPECT_EQ(a.ifdForPlane(0, 0, 0), 4u) << "the prefixed <TiffData> must map its planes";
	EXPECT_EQ(a.ifdForPlane(1, 1, 0), 7u);

	// and a prefixed companion block is still caught
	const std::string companion =
		"<ome:OME xmlns:ome=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">"
		"<ome:Image><ome:Pixels ID=\"p\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
		"SizeX=\"8\" SizeY=\"6\" SizeZ=\"2\" SizeC=\"2\" SizeT=\"1\">"
		"<ome:TiffData IFD=\"0\" PlaneCount=\"4\">"
		"<ome:UUID FileName=\"other.ome.tif\">urn:uuid:b</ome:UUID></ome:TiffData>"
		"</ome:Pixels></ome:Image></ome:OME>";

	Nyxus::OmeAxes c = Nyxus::parse_ome_xml(companion, "self.ome.tif");
	ASSERT_TRUE(c.valid);
	EXPECT_TRUE(c.multiFileTiff);
}

// A prefix is a name, so "<omePixels" or a prefix carrying a space is not a prefixed <Pixels>.
TEST(OmeTiffMetaBad, PrefixMustBeAName)
{
	EXPECT_FALSE(Nyxus::parse_ome_xml("<OME><omePixels SizeX=\"8\" SizeY=\"6\"/></OME>").valid);
	EXPECT_FALSE(Nyxus::parse_ome_xml("<OME><a:b:Pixels SizeX=\"8\" SizeY=\"6\"/></OME>").valid);
	EXPECT_FALSE(Nyxus::parse_ome_xml("<OME><Image ome:Pixels=\"no\"/></OME>").valid);
}

TEST(OmeTiffMeta, Parses2DSingletonAxesCollapse)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(kOmeXml_2d);
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeZ, 1u);
	EXPECT_EQ(a.sizeC, 1u);
	EXPECT_EQ(a.sizeT, 1u);
	EXPECT_EQ(a.storageOrder, "YX");     // singleton Z/C/T dropped; planes stay YX
}

TEST(OmeTiffMeta, InvalidWhenNoPixels)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<OME><Image/></OME>");
	EXPECT_FALSE(a.valid);
}

// ---------------------------------------------------------------------------
// Negative / robustness: bad or missing metadata for every field we parse.
// The parser must never crash and must degrade to safe defaults.
// ---------------------------------------------------------------------------

TEST(OmeTiffMetaBad, EmptyAndGarbageInput)
{
	EXPECT_FALSE(Nyxus::parse_ome_xml("").valid);
	EXPECT_FALSE(Nyxus::parse_ome_xml("not xml at all").valid);
	EXPECT_FALSE(Nyxus::parse_ome_xml("<OME></OME>").valid);
}

TEST(OmeTiffMetaBad, UnterminatedPixelsTagIsInvalid)
{
	// no '>' closing the Pixels start-tag
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<OME><Image><Pixels SizeX=\"8\" SizeY=\"6\"");
	EXPECT_FALSE(a.valid);
}

TEST(OmeTiffMetaBad, PixelsNamePrefixDoesNotFalseMatch)
{
	// "<PixelsFoo" must not be taken as "<Pixels"
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<OME><PixelsFoo SizeX=\"8\"/></OME>");
	EXPECT_FALSE(a.valid);
}

TEST(OmeTiffMetaBad, AllSizesMissingDefaultToOne)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<OME><Image><Pixels ID=\"p\"></Pixels></Image></OME>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 1u); EXPECT_EQ(a.sizeY, 1u); EXPECT_EQ(a.sizeZ, 1u);
	EXPECT_EQ(a.sizeC, 1u); EXPECT_EQ(a.sizeT, 1u);
	EXPECT_EQ(a.storageOrder, "YX");
	EXPECT_EQ(a.omeDimensionOrder, "XYZCT");     // missing -> default
	EXPECT_EQ(a.dtype, Nyxus::PixelType::UInt16); // missing Type -> default
	EXPECT_DOUBLE_EQ(a.physX, 1.0);               // missing PhysicalSize -> 1.0
	EXPECT_DOUBLE_EQ(a.physZ, 1.0);
	EXPECT_TRUE(a.unitXY.empty());
	EXPECT_TRUE(a.unitZ.empty());
}

TEST(OmeTiffMetaBad, NonNumericSizeDefaultsToOne)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels SizeX=\"abc\" SizeY=\"6\" SizeZ=\"\" SizeC=\"1\" SizeT=\"1\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 1u);   // "abc" -> 1, not 0
	EXPECT_EQ(a.sizeY, 6u);
	EXPECT_EQ(a.sizeZ, 1u);   // "" -> 1
}

TEST(OmeTiffMetaBad, ZeroSizeClampedToOne)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels SizeX=\"0\" SizeY=\"0\" SizeZ=\"4\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 1u);   // 0 would divide-by-zero the tile grid
	EXPECT_EQ(a.sizeY, 1u);
	EXPECT_EQ(a.sizeZ, 4u);
}

TEST(OmeTiffMetaBad, NegativeSizeRejectedToDefault)
{
	// strtoull wraps a leading '-' into a huge extent; parser must reject it
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<Pixels SizeX=\"-5\" SizeY=\"6\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 1u);        // negative -> default 1, not a giant number
	EXPECT_EQ(a.sizeY, 6u);
}

TEST(OmeTiffMetaBad, MissingDimensionOrderDefaults)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml("<Pixels SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\"></Pixels>");
	EXPECT_EQ(a.omeDimensionOrder, "XYZCT");
	EXPECT_EQ(a.storageOrder, "ZYX");
}

TEST(OmeTiffMetaBad, MalformedDimensionOrderFallsBack)
{
	for (const char* bad : { "FOOBAR", "XYZC", "XYZCC", "XXYZCT", "xyzct", "" })
	{
		std::string xml = std::string("<Pixels DimensionOrder=\"") + bad +
			"\" SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\"></Pixels>";
		Nyxus::OmeAxes a = Nyxus::parse_ome_xml(xml);
		ASSERT_TRUE(a.valid) << bad;
		EXPECT_EQ(a.omeDimensionOrder, "XYZCT") << "bad input: " << bad;
	}
}

TEST(OmeTiffMetaBad, SingleQuotedAttributes)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels DimensionOrder='XYZCT' Type='uint8' SizeX='8' SizeY='6' SizeZ='4' PhysicalSizeX='0.25'></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.sizeZ, 4u);
	EXPECT_EQ(a.dtype, Nyxus::PixelType::UInt8);
	EXPECT_DOUBLE_EQ(a.physX, 0.25);
}

TEST(OmeTiffMetaBad, UnknownAndMissingTypeFallBackToUint16)
{
	EXPECT_EQ(Nyxus::parse_ome_xml("<Pixels Type=\"bogus\" SizeX=\"8\" SizeY=\"6\"></Pixels>").dtype,
		Nyxus::PixelType::UInt16);
	EXPECT_EQ(Nyxus::parse_ome_xml("<Pixels SizeX=\"8\" SizeY=\"6\"></Pixels>").dtype,
		Nyxus::PixelType::UInt16);
}

TEST(OmeTiffMetaBad, AllPixelTypesResolve)
{
	auto ty = [](const char* t) {
		std::string xml = std::string("<Pixels Type=\"") + t + "\" SizeX=\"8\" SizeY=\"6\"></Pixels>";
		return Nyxus::parse_ome_xml(xml).dtype;
	};
	EXPECT_EQ(ty("uint8"), Nyxus::PixelType::UInt8);
	EXPECT_EQ(ty("int16"), Nyxus::PixelType::Int16);
	EXPECT_EQ(ty("int32"), Nyxus::PixelType::Int32);
	EXPECT_EQ(ty("float"), Nyxus::PixelType::Float32);
	EXPECT_EQ(ty("double"), Nyxus::PixelType::Float64);
	EXPECT_EQ(Nyxus::parse_ome_xml("<Pixels Type=\"double\" SizeX=\"8\" SizeY=\"6\"></Pixels>").bitsPerSample, 64u);
}

TEST(OmeTiffMetaBad, NonNumericPhysicalSizesDefaultToOne)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\" SizeT=\"2\" "
		"PhysicalSizeX=\"abc\" PhysicalSizeZ=\"\" TimeIncrement=\"NaNsense\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_DOUBLE_EQ(a.physX, 1.0);          // failed parse -> default, never 0
	EXPECT_DOUBLE_EQ(a.physZ, 1.0);
}

TEST(OmeTiffMetaBad, NanInfPhysicalSizesRejected)
{
	// strtod parses "NaN"/"inf" prefixes; such values must not leak into calibration
	for (const char* bad : { "NaN", "NaNsense", "inf", "-inf", "infinity" })
	{
		std::string xml = std::string("<Pixels SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\" PhysicalSizeZ=\"") +
			bad + "\" TimeIncrement=\"" + bad + "\"></Pixels>";
		Nyxus::OmeAxes a = Nyxus::parse_ome_xml(xml);
		ASSERT_TRUE(a.valid) << bad;
		EXPECT_DOUBLE_EQ(a.physZ, 1.0) << "PhysicalSizeZ from " << bad;
	}
}

TEST(OmeTiffMetaBad, SizeXNotConfusedWithPhysicalSizeX)
{
	// attribute-name boundary: SizeX must not read from PhysicalSizeX
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels PhysicalSizeX=\"9.9\" PhysicalSizeXUnit=\"nm\" SizeX=\"8\" SizeY=\"6\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	// physX/unitXY are canonicalized to micrometer (9.9 nm = 0.0099 um); see
	// UnitCanonicalization.NanometerConvertedToMicrometer for a test dedicated to that.
	EXPECT_DOUBLE_EQ(a.physX, 0.0099);
	EXPECT_EQ(a.unitXY, "micrometer");
}

// canonicalize_to_micrometer / unit_scale_to_micrometer directly -- no XML/JSON parsing
// involved, so these pin the conversion table itself independent of either format parser.
TEST(UnitCanonicalization, KnownUnitsConvertedToMicrometer)
{
	struct Case { const char* unit; double input; double expected_um; };
	const Case cases[] = {
		{ "nanometer", 500.0, 0.5 },
		{ "nm",        500.0, 0.5 },
		{ "micrometer", 2.0,  2.0 },
		{ "um",         2.0,  2.0 },
		{ "millimeter", 0.01, 10.0 },
		{ "mm",         0.01, 10.0 },
		{ "centimeter", 0.001, 10.0 },
		{ "meter",      0.000002, 2.0 },
		{ "angstrom",   50000.0, 5.0 },
	};
	for (const auto& c : cases)
	{
		double v = c.input;
		std::string u = c.unit;
		Nyxus::canonicalize_to_micrometer(v, u);
		EXPECT_NEAR(v, c.expected_um, 1e-9) << "unit=" << c.unit;
		EXPECT_EQ(u, "micrometer") << "unit=" << c.unit;
	}
}

TEST(UnitCanonicalization, UnrecognizedOrEmptyUnitLeftAsIs)
{
	for (const char* unit : { "", "furlong", "pixel" })
	{
		double v = 42.0;
		std::string u = unit;
		Nyxus::canonicalize_to_micrometer(v, u);
		EXPECT_DOUBLE_EQ(v, 42.0) << "unit=" << unit;
		EXPECT_EQ(u, unit) << "unit=" << unit;
	}
}

TEST(UnitCanonicalization, OmeTiffNanometerXAndMillimeterZBothConvert)
{
	// X/Y declared in nanometer, Z in a DIFFERENT unit (millimeter) -- each axis must
	// convert using its OWN declared unit, not X/Y's unit applied to Z (or vice versa).
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels SizeX=\"8\" SizeY=\"6\" SizeZ=\"4\" "
		"PhysicalSizeX=\"500\" PhysicalSizeXUnit=\"nanometer\" "
		"PhysicalSizeY=\"500\" "
		"PhysicalSizeZ=\"0.003\" PhysicalSizeZUnit=\"millimeter\"></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_NEAR(a.physX, 0.5, 1e-9);
	EXPECT_NEAR(a.physY, 0.5, 1e-9);
	EXPECT_NEAR(a.physZ, 3.0, 1e-9);
	EXPECT_EQ(a.unitXY, "micrometer");
	EXPECT_EQ(a.unitZ, "micrometer");
	// storageAxes mirror the canonicalized per-axis values too
	int ix = a.storageIndexOf('X'), iy = a.storageIndexOf('Y'), iz = a.storageIndexOf('Z');
	ASSERT_GE(ix, 0); ASSERT_GE(iy, 0); ASSERT_GE(iz, 0);
	EXPECT_NEAR(a.storageAxes[ix].physical, 0.5, 1e-9);
	EXPECT_NEAR(a.storageAxes[iy].physical, 0.5, 1e-9);
	EXPECT_NEAR(a.storageAxes[iz].physical, 3.0, 1e-9);
}

TEST(OmeTiffMetaBad, WhitespaceAndNewlinesInTag)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels\n\t DimensionOrder = \"XYZCT\"\n  SizeX=\"8\"\r\n SizeY=\"6\"  SizeZ=\"4\" ></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.sizeZ, 4u);
	EXPECT_EQ(a.omeDimensionOrder, "XYZCT");
}

TEST(OmeTiffMetaBad, ChannelElementsDoNotChangeSizeC)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<Pixels SizeX=\"8\" SizeY=\"6\" SizeC=\"2\">"
		"<Channel ID=\"c0\" SamplesPerPixel=\"1\"/><Channel ID=\"c1\"/></Pixels>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeC, 2u);
}

TEST(OmeTiffMetaBad, FirstPixelsWinsAcrossMultipleImages)
{
	Nyxus::OmeAxes a = Nyxus::parse_ome_xml(
		"<OME><Image><Pixels SizeX=\"8\" SizeY=\"6\"></Pixels></Image>"
		"<Image><Pixels SizeX=\"64\" SizeY=\"64\"></Pixels></Image></OME>");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.sizeY, 6u);
}

TEST(OmeAxes, PixelTypeMappings)
{
	using namespace Nyxus;
	EXPECT_EQ(pixel_type_from_ome_type("uint16"), PixelType::UInt16);
	EXPECT_EQ(pixel_type_from_ome_type("float"), PixelType::Float32);
	EXPECT_EQ(pixel_type_from_ome_type("double"), PixelType::Float64);
	// zarr v2 numpy strings and v3 names both resolve
	EXPECT_EQ(pixel_type_from_zarr_dtype("<u2"), PixelType::UInt16);
	EXPECT_EQ(pixel_type_from_zarr_dtype("uint16"), PixelType::UInt16);
	EXPECT_EQ(pixel_type_from_zarr_dtype("<i4"), PixelType::Int32);
	EXPECT_EQ(pixel_type_from_zarr_dtype("float64"), PixelType::Float64);
	EXPECT_EQ(bits_of(PixelType::UInt16), 16u);
	EXPECT_TRUE(is_float(PixelType::Float32));
	// OME DimensionOrder synthesis from an on-disk order
	EXPECT_EQ(ome_dimension_order_from_storage("TCZYX"), "XYZCT");
	EXPECT_EQ(ome_dimension_order_from_storage("CTZYX"), "XYZTC");
	EXPECT_EQ(ome_dimension_order_from_storage("ZYX"), "XYZCT");
}

#ifdef OMEZARR_SUPPORT
#include "../src/nyx/ome/ome_zarr_meta.h"

TEST(OmeZarrMeta, ParsesNgffAxesAndPyramid)
{
	// NGFF 0.4-style group attrs with a non-default C/T order + 2 pyramid levels.
	nlohmann::json attrs = nlohmann::json::parse(R"({
		"multiscales":[{"version":"0.4",
			"axes":[{"name":"c","type":"channel"},{"name":"t","type":"time","unit":"second"},
			        {"name":"z","type":"space","unit":"micrometer"},
			        {"name":"y","type":"space","unit":"micrometer"},
			        {"name":"x","type":"space","unit":"micrometer"}],
			"datasets":[
				{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,1,2.0,0.5,0.5]}]},
				{"path":"1","coordinateTransformations":[{"type":"scale","scale":[1,1,2.0,1.0,1.0]}]}]}]})");
	std::vector<std::size_t> shape = {3, 2, 4, 6, 8};   // C,T,Z,Y,X
	Nyxus::OmeAxes a = Nyxus::parse_ome_zarr(attrs, shape, "<u2");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeC, 3u);
	EXPECT_EQ(a.sizeT, 2u);
	EXPECT_EQ(a.sizeZ, 4u);
	EXPECT_EQ(a.sizeY, 6u);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.storageOrder, "CTZYX");
	EXPECT_EQ(a.omeDimensionOrder, "XYZTC");
	EXPECT_EQ(a.numberPyramidLevels(), 2u);
	EXPECT_DOUBLE_EQ(a.physZ, 2.0);
	EXPECT_DOUBLE_EQ(a.physX, 0.5);
	EXPECT_EQ(a.dtype, Nyxus::PixelType::UInt16);
}

// ---- OME-Zarr negative / robustness ----

static Nyxus::OmeAxes zarr_parse(const char* js, std::vector<std::size_t> shape, const char* dt = "<u2")
{
	return Nyxus::parse_ome_zarr(nlohmann::json::parse(js), shape, dt);
}

TEST(OmeZarrMetaBad, InvalidWhenStructureMissing)
{
	EXPECT_FALSE(zarr_parse(R"({})", {6, 8}).valid);                              // no multiscales
	EXPECT_FALSE(zarr_parse(R"({"multiscales":[]})", {6, 8}).valid);             // empty list
	EXPECT_FALSE(zarr_parse(R"({"multiscales":[{"datasets":[{"path":"0"}]}]})", {6, 8}).valid);  // no axes
	EXPECT_FALSE(zarr_parse(R"({"multiscales":[{"axes":[{"name":"y","type":"space"}]}]})", {6}).valid); // no datasets
	EXPECT_FALSE(zarr_parse(R"({"multiscales":[{"axes":[{"name":"y","type":"space"}],"datasets":[]}]})", {6}).valid); // empty datasets
}

TEST(OmeZarrMetaBad, DatasetWithoutCoordinateTransformationsDoesNotThrow)
{
	// The pre-hardening code did `datasets[0]["coordinateTransformations"]` which
	// throws on a missing key. Must now default scale to 1.0 and stay valid.
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":[{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0"}]}]})", {6, 8});
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeY, 6u);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_DOUBLE_EQ(a.physX, 1.0);
	EXPECT_DOUBLE_EQ(a.physY, 1.0);
	EXPECT_EQ(a.numberPyramidLevels(), 1u);
}

TEST(OmeZarrMetaBad, ScaleLengthMismatchDefaultsToOne)
{
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":[{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,2,3]}]}]}]})",
		{6, 8});
	ASSERT_TRUE(a.valid);
	EXPECT_DOUBLE_EQ(a.physX, 1.0);   // 3 scales vs 2 axes -> all default
	EXPECT_DOUBLE_EQ(a.physY, 1.0);
}

TEST(OmeZarrMetaBad, NonNumericScaleElementDefaultsThatAxis)
{
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":[{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":["oops",0.5]}]}]}]})",
		{6, 8});
	ASSERT_TRUE(a.valid);
	EXPECT_DOUBLE_EQ(a.physY, 1.0);   // "oops" -> 1.0
	EXPECT_DOUBLE_EQ(a.physX, 0.5);
}

TEST(OmeZarrMetaBad, MalformedAxisEntrySkipped)
{
	// a non-object axis entry must be skipped, not crash
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":["notanobject",{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,0.5,0.5]}]}]}]})",
		{1, 6, 8});
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeX, 8u);
	EXPECT_EQ(a.sizeY, 6u);
}

TEST(OmeZarrMetaBad, UnknownDtypeFallsBackUint16)
{
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":[{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,1]}]}]}]})",
		{6, 8}, "weird64");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.dtype, Nyxus::PixelType::UInt16);
}

TEST(OmeZarrMetaBad, V3OmeKeyPathParses)
{
	// 0.5 nests the model under "ome"
	Nyxus::OmeAxes a = zarr_parse(
		R"({"ome":{"version":"0.5","multiscales":[{
			"axes":[{"name":"z","type":"space"},{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[2.0,0.5,0.5]}]}]}]}})",
		{4, 6, 8}, "uint16");
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeZ, 4u);
	EXPECT_EQ(a.storageOrder, "ZYX");
	EXPECT_DOUBLE_EQ(a.physZ, 2.0);
}

TEST(OmeZarrMetaBad, ShorterShapeThanAxesDefaultsMissingSizes)
{
	// level0Shape shorter than axes list -> missing dims default to size 1, no OOB
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{"axes":[{"name":"c","type":"channel"},{"name":"y","type":"space"},{"name":"x","type":"space"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[1,0.5,0.5]}]}]}]})",
		{3});   // only C provided
	ASSERT_TRUE(a.valid);
	EXPECT_EQ(a.sizeC, 3u);
	EXPECT_EQ(a.sizeY, 1u);
	EXPECT_EQ(a.sizeX, 1u);
}

TEST(UnitCanonicalization, OmeZarrNanometerXYAndAngstromZBothConvert)
{
	// X/Y declared in nanometer, Z in a DIFFERENT unit (angstrom) -- each axis must
	// convert using its OWN declared unit.
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{
			"axes":[{"name":"z","type":"space","unit":"angstrom"},
			        {"name":"y","type":"space","unit":"nanometer"},
			        {"name":"x","type":"space","unit":"nanometer"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[30000.0,500.0,500.0]}]}]}]})",
		{4, 6, 8}, "uint16");
	ASSERT_TRUE(a.valid);
	EXPECT_NEAR(a.physX, 0.5, 1e-9);
	EXPECT_NEAR(a.physY, 0.5, 1e-9);
	EXPECT_NEAR(a.physZ, 3.0, 1e-9);
	EXPECT_EQ(a.unitXY, "micrometer");
	EXPECT_EQ(a.unitZ, "micrometer");
}

TEST(UnitCanonicalization, OmeZarrAlreadyMicrometerUnaffected)
{
	// An already-canonical store must come through byte-for-byte unchanged (no accidental
	// double-conversion).
	Nyxus::OmeAxes a = zarr_parse(
		R"({"multiscales":[{
			"axes":[{"name":"z","type":"space","unit":"micrometer"},
			        {"name":"y","type":"space","unit":"micrometer"},
			        {"name":"x","type":"space","unit":"micrometer"}],
			"datasets":[{"path":"0","coordinateTransformations":[{"type":"scale","scale":[2.0,0.5,0.5]}]}]}]})",
		{4, 6, 8}, "uint16");
	ASSERT_TRUE(a.valid);
	EXPECT_DOUBLE_EQ(a.physX, 0.5);
	EXPECT_DOUBLE_EQ(a.physY, 0.5);
	EXPECT_DOUBLE_EQ(a.physZ, 2.0);
	EXPECT_EQ(a.unitXY, "micrometer");
	EXPECT_EQ(a.unitZ, "micrometer");
}
#endif
