#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "histogram.h"
#include "sensemaker.h"
#include "version.h"

bool test_histogram()
{
	auto h = std::make_shared <Histo>();
	h->add_observation(23);
	h->add_observation(150);
	h->add_observation(58);
	h->add_observation(32);
	h->add_observation(75);
	h->add_observation(2);		h->print("add 2");
	h->add_observation(208);
	h->add_observation(208);	h->print("add 208");
	h->add_observation(34);	h->print("add 34");
	h->add_observation(43);
	h->add_observation(99);
	h->add_observation(11);
	h->add_observation(64);
	h->add_observation(37);
	h->add_observation(23);
	h->add_observation(151);
	h->add_observation(49);
	h->add_observation(88);
	h->add_observation(55);	h->print("add 55");
	h->add_observation(0);		h->print("add 0");
	h->add_observation(77);	h->print("add 77");
	h->add_observation(38);
	h->add_observation(33);
	h->add_observation(21);
	h->add_observation(61);
	h->print();
	h->reset();
	return true;
}
