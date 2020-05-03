
- we have N alerts
- we have P properties
- alerts get p properties (p <= P)

- we define zones : all properties associated with an object define a zone

- we can define a distance between zones:
  - number of properties NOT in common to the two zones (``symmetric_difference``)
  - weighted by the sum of properties number of the two zones

- we can compute the distance between two objects = distance of their zones
- we can get all shortest neighbours of an object (out of its zone of course)
  - those are interesting neighbours

- we set a link between two objects, when their zones is apart by a given distance

